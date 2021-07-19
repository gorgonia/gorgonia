package exprgraph_test

import (
	"fmt"

	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
type FwdEngine struct {
	tensor.StdEng
	g *exprgraph.Graph
}

func (e *FwdEngine) Graph() *exprgraph.Graph { return e.g }

func (e *FwdEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *FwdEngine) Lift(a exprgraph.Tensor) exprgraph.Tensor {
	switch t := a.(type) {
	case *dual.Dual:
		return a
	case tensor.Tensor:
		return dual.New(t)
	}
	panic("Unreachable")
}

func (e *FwdEngine) MatMul(a, b, c tensor.Tensor) error {
	adv := a.(*dual.Dual)
	bdv := b.(*dual.Dual)
	cdv := c.(*dual.Dual)

	if err := e.StdEng.MatMul(adv.Value, bdv.Value, cdv.Value); err != nil {
		return err
	}

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	if err := bdv.Value.T(); err != nil {
		return err // cannot transpose
	}

	// dA = C×B'
	if err := e.StdEng.MatMul(cdv.Value, bdv.Value, advd); err != nil {
		return err
	}

	if err := adv.Value.T(); err != nil {
		return err // cannot transpose
	}

	// dB = A'×C
	if err := e.StdEng.MatMul(adv.Value, cdv.Value, bdvd); err != nil {
		return err
	}

	// now we undo our transposes
	adv.Value.UT()
	bdv.Value.UT()

	return nil
}

func (e *FwdEngine) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	adv := a.(*dual.Dual)
	bdv := b.(*dual.Dual)
	c, err := e.StdEng.AddScalar(a, b, leftTensor, opts...)
	if err != nil {
		return nil, err
	}
	c = e.Lift(c).(tensor.Tensor)

	advd := adv.Deriv()
	bdvd := bdv.Deriv()
	advd.Memset(1.0) // this is assuming we only work in float64. More needs to be done here
	bdvd.Memset(1.0)
	return c, nil
}

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
func Example_forward_differentiation_engine() {
	engine := &FwdEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))

	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Println(err)
	}

	getDeriv := func(t gorgonia.Tensor) values.Value {
		n := t.(*exprgraph.Node)
		return n.Tensor.(*dual.Dual).Deriv()
	}

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)
	fmt.Printf("dx:\n%v\ndy:\n%v\ndxy:\n%v\ndxy+z:\n%v\n", getDeriv(x), getDeriv(y), getDeriv(xy), getDeriv(xypz))

	// Output:
	// x:
	// ⎡1  2  3⎤
	// ⎣4  5  6⎦
	//
	// y:
	// ⎡6  5⎤
	// ⎢4  3⎥
	// ⎣2  1⎦
	//
	// xy:
	// ⎡20  14⎤
	// ⎣56  41⎦
	//
	// xy+z:
	// ⎡21  15⎤
	// ⎣57  42⎦
	//
	// dx:
	// ⎡190  122   54⎤
	// ⎣541  347  153⎦
	//
	// dy:
	// ⎡244  178⎤
	// ⎢320  233⎥
	// ⎣396  288⎦
	//
	// dxy:
	// ⎡1  1⎤
	// ⎣1  1⎦
	//
	// dxy+z:
	// ⎡0  0⎤
	// ⎣0  0⎦

}
