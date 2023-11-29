package exprgraph_test

import (
	"context"
	"fmt"
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

var (
	_ tensor.BLA[float64, *dense.Dense[float64]] = &BwdEngine[float64, *dense.Dense[float64]]{}
)

type adInstr[DT any, T tensor.Tensor[DT, T]] struct {
	ADOp[DT, T]

	inputs []gorgonia.Tensor
	output gorgonia.Tensor
}

func (ad adInstr[DT, T]) do(ctx context.Context) error { return ad.DoDiff(ctx, ad.inputs, ad.output) }

// BwdEngine is an Engine that performs backwards mode diffentiation.
type BwdEngine[DT tensor.Num, T tensor.Tensor[DT, T]] struct {
	dense.StdFloat64Engine[*dense.Dense[float64]]
	g *exprgraph.Graph

	q []adInstr[DT, T]
}

func (e *BwdEngine[DT, T]) Graph() *exprgraph.Graph { return e.g }

func (e *BwdEngine[DT, T]) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *BwdEngine[DT, T]) Lift(a exprgraph.Tensor) exprgraph.Tensor {
	switch t := a.(type) {
	case *dual.Dual[DT, T]:
		return a
	case T:
		return dual.New[DT, T](t)
	}
	panic("Unreachable")
}

func (e *BwdEngine[DT, T]) Inner(ctx context.Context, a, b T) (DT, error) {
	return 0, errors.New("NYI")
}

func (e *BwdEngine[DT, T]) FMA(ctx context.Context, a, x, retVal T) error {
	return errors.New("NYI")
}

func (e *BwdEngine[DT, T]) MatVecMul(ctx context.Context, a, b, retVal T, incr []DT) error {
	return errors.New("NYI")
}

func (e *BwdEngine[DT, T]) Outer(ctx context.Context, a, b, retVal T, incr []DT) error {
	return errors.New("NYI")
}

func (e *BwdEngine[DT, T]) MatMul(ctx context.Context, a, b, c T, incr []DT) error {
	mm, ok := e.Engine.(tensor.BLA[DT, T])
	if !ok {
		return errors.New("Expected BLA")
	}
	err := mm.MatMul(ctx, a, b, c, incr)
	log.Printf("err in MatMul %v", err)
	return err
}

/*
func (e *BwdEngine[DT, T]) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
	fo := tensor.ParseFuncOpts(opts...)
	reuse := fo.Reuse()

	var cdv *dual.Dual
	if reuse != nil {
		switch rt := reuse.(type) {
		case *dual.Dual:
			cdv = rt
			fo.SetReuse(cdv.Value)
		case tensor.Tensor:
			fo.SetReuse(rt)
		}

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
	return c, nil
        }
*/

func (e *BwdEngine[DT, T]) Q(op ops.Op[DT, T], inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	var ad ADOp[DT, T]
	var ok bool
	if ad, ok = op.(ADOp[DT, T]); !ok {
		return errors.Errorf("Expected %v to be an ADOp", op)
	}
	e.q = append(e.q, adInstr[DT, T]{ad, inputs, output})
	return nil
}

func (e *BwdEngine[DT, T]) Backwards(ctx context.Context) error {
	// TODO: pickup custom gradients
	for _, ad := range e.q {
		if err := ad.do(ctx); err != nil {
			return err
		}
	}
	return nil
}

func Example_backward_differentiation_engine() {
	engine := &BwdEngine[float64, *dense.Dense[float64]]{Engine: dense.StdFloat64Engine[*dense.Dense[float64]]{}}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	_, ok := g.Engine.(tensor.BLA[float64, *dense.Dense[float64]])
	log.Printf("g is a BLA? ok %v", ok)

	x := exprgraph.New[float64](g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.New[float64](g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.New[float64](g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul[float64, *dense.Dense[float64]](x, y)
	if err != nil {
		fmt.Printf("Matmul failed: Err: %v\n", err)
		return
	}
	log.Printf("xy %v xy==nil %v | %v", xy, xy == nil, err)

	xypz, err := Add[float64, *dense.Dense[float64]](xy, z)
	if err != nil {
		fmt.Printf("Add failed. Err: %v\n", err)
		return
	}

	if err := engine.Backwards(nil); err != nil {
		fmt.Printf("Backwards failed. Err: %v\n", err)
		return
	}

	// note: getDeriv is defined in example_utils_test.go
	getD := getDeriv[float64, *dense.Dense[float64]]

	fmt.Printf("x:\n%v\ny:\n%v\nxy:\n%v\nxy+z:\n%v\n", x, y, xy, xypz)
	fmt.Printf("dx:\n%v\ndy:\n%v\ndxy:\n%v\ndxy+z:\n%v\n", getD(x), getD(y), getD(xy), getD(xypz))

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

/*
func Example_backward_differentiation_engine_samenodename() {
	engine := &BwdEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	x2 := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{100, 200, 300, 400, 500, 600}))

	fmt.Printf("%v\n%v", x, x2)

	// Output:

}
*/
