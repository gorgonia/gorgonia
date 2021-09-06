package exprgraph_test

import (
	"context"
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

type adInstr struct {
	ADOp

	inputs []gorgonia.Tensor
	output gorgonia.Tensor
}

func (ad adInstr) do(ctx context.Context) error { return ad.DoDiff(ctx, ad.inputs, ad.output) }

// BwdEngine is an Engine that performs backwards mode diffentiation.
type BwdEngine struct {
	tensor.StdEng
	g *exprgraph.Graph

	q []adInstr
}

func (e *BwdEngine) Graph() *exprgraph.Graph { return e.g }

func (e *BwdEngine) SetGraph(g *exprgraph.Graph) { e.g = g }

func (e *BwdEngine) Lift(a exprgraph.Tensor) exprgraph.Tensor {
	switch t := a.(type) {
	case *dual.Dual:
		return a
	case tensor.Tensor:
		return dual.New(t)
	}
	panic("Unreachable")
}

func (e *BwdEngine) MatMul(ctx context.Context, a, b, c tensor.Tensor) error {
	var av, bv, cv tensor.Tensor
	switch at := a.(type) {
	case *dual.Dual:
		av = at.Value
	case tensor.Tensor:
		av = at
	}

	switch bt := b.(type) {
	case *dual.Dual:
		bv = bt.Value
	case tensor.Tensor:
		bv = bt
	}

	switch ct := c.(type) {
	case *dual.Dual:
		cv = ct.Value
	case tensor.Tensor:
		cv = ct
	}

	return e.StdEng.MatMul(ctx, av, bv, cv)
}

func (e *BwdEngine) AddScalar(a tensor.Tensor, b interface{}, leftTensor bool, opts ...tensor.FuncOpt) (tensor.Tensor, error) {
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

func (e *BwdEngine) Q(op ops.Op, inputs []gorgonia.Tensor, output gorgonia.Tensor) error {
	var ad ADOp
	var ok bool
	if ad, ok = op.(ADOp); !ok {
		return errors.Errorf("Expected %v to be an ADOp", op)
	}
	e.q = append(e.q, adInstr{ad, inputs, output})
	return nil
}

func (e *BwdEngine) Backwards(ctx context.Context) error {
	for _, ad := range e.q {
		if err := ad.do(ctx); err != nil {
			return err
		}
	}
	return nil
}

func Example_backward_differentiation_engine() {
	engine := &BwdEngine{}
	g := exprgraph.NewGraph(engine)
	engine.g = g

	x := exprgraph.NewNode(g, "x", tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	y := exprgraph.NewNode(g, "y", tensor.WithShape(3, 2), tensor.WithBacking([]float64{6, 5, 4, 3, 2, 1}))
	z := exprgraph.NewNode(g, "z", tensor.WithShape(), tensor.WithBacking([]float64{1}))
	xy, err := MatMul(x, y)
	if err != nil {
		fmt.Printf("Matmul failed: Err: %v\n", err)
		return
	}

	xypz, err := Add(xy, z)
	if err != nil {
		fmt.Printf("Add failed. Err: %v\n", err)
		return
	}

	if err := engine.Backwards(nil); err != nil {
		fmt.Printf("Backwards failed. Err: %v\n", err)
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
