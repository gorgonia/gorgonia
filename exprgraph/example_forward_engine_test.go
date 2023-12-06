package exprgraph_test

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
type FwdEngine[DT tensor.Num, T tensor.Tensor[DT, T]] struct {
	StandardEngine[DT, T]
	g *exprgraph.Graph
}

func (e *FwdEngine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *FwdEngine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *FwdEngine[DT, T]) Lift(a exprgraph.Tensor) exprgraph.Tensor {
	switch t := a.(type) {
	case *dual.Dual[DT, T]:
		return a
	case T:
		return dual.New[DT, T](t)
	}
	panic("Unreachable")
}

func (e *FwdEngine[DT, T]) Inner(ctx context.Context, a, b T) (DT, error) {
	return 0, errors.New("NYI")
}

func (e *FwdEngine[DT, T]) FMA(ctx context.Context, a, x, retVal T) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatVecMul(ctx context.Context, a, b, retVal T, incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) Outer(ctx context.Context, a, b, retVal T, incr []DT) error {
	return errors.New("NYI")
}

func (e *FwdEngine[DT, T]) MatMul(ctx context.Context, a, b, c T, incr []DT) error {
	adv := a.(*dual.Dual[DT, T])
	bdv := b.(*dual.Dual[DT, T])
	cdv := c.(*dual.Dual[DT, T])

	if err := e.StandardEngine.MatMul(ctx, adv.Value, bdv.Value, cdv.Value); err != nil {
		return err
	}

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	if err := bdv.Value.T(); err != nil {
		return err // cannot transpose
	}

	// dA = C×B'
	if err := e.StandardEngine.MatMul(ctx, cdv.Value, bdv.Value, advd); err != nil {
		return err
	}

	if err := adv.Value.T(); err != nil {
		return err // cannot transpose
	}

	// dB = A'×C
	if err := e.StandardEngine.MatMul(ctx, adv.Value, cdv.Value, bdvd); err != nil {
		return err
	}

	// now we undo our transposes
	adv.Value.UT()
	bdv.Value.UT()

	return nil
}

/*
func (e *FwdEngine[DT, T]) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	adv := a.(*dual.Dual)
	bdv := b.(*dual.Dual)
	fo := tensor.ParseFuncOpts(opts...)
	reuse := fo.Reuse()

	var cdv *dual.Dual
	if reuse != nil {
		cdv = reuse.(*dual.Dual)
		fo.SetReuse(cdv.Value)
	}

	c, err := e.StdEng.AddScalar(a, b, leftTensor, fo.FuncOpts()...)
	if err != nil {
		return nil, err
	}
	if cdv == nil {
		c = e.Lift(c).(tensor.Tensor)
	} else {
		cdv.Value = c
		c = cdv
	}

	advd := adv.Deriv()
	bdvd := bdv.Deriv()
	advd.Memset(1.0) // this is assuming we only work in float64. More needs to be done here
	bdvd.Memset(1.0)
	return c, nil
}
*/
// FwdEngine is a Engine that performs forwards mode differentiation
//
// Here the implementation is done by means of implementing MatMul and AddScalar
// Obviously in the real world situation, Add also needs to be implemented, but in this example
// we are not going to call Add, only AddScalar.
func Example_forward_differentiation_engine() {
	engine := &FwdEngine[float64, *dense.Dense[float64]]{StandardEngine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul[float64](x, y)
	if err != nil {
		fmt.Println(err)
	}

	xypz, err := Add[float64](xy, z)
	if err != nil {
		fmt.Println(err)
		return
	}

	// note: getDeriv is defined in example_utils_test.go

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
