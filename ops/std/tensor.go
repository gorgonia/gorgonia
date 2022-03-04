package stdops

import (
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// this file provides the standard tensor ops

/*
   These are operations that are "native" to what a tensor can do:
   	- At
        - Slice
        - Transpose
        - SizeOf (Index in shapes package)

*/

// At represents a coordinate to get a value from a tensor
type At []int

func (op At) Arity() int { return 1 }

func (op At) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.TensorType{Dims: len(op), Of: a}
	return hm.NewFnType(t, a)
}

func (op At) ShapeExpr() shapes.Expr { return shapes.MakeArrow(shapes.Var('a'), shapes.ScalarShape()) } // TODO: leverage shape package to actually add more checks

func (op At) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	v := vs[0].(tensor.Tensor)
	_, task := trace.NewTask(ctx, op.String())
	var r interface{}
	if r, err = v.At(op...); err == nil {
		retVal = values.MakeScalar(r)
	}
	task.End()
	return retVal, err

}

func (op At) String() string { return fmt.Sprintf("At(%v)", []int(op)) }

// Size is an operation that gets the size of a given axis in a shape. It returns the size in the datatype of the input
type Size int

func (op Size) Arity() int { return 1 }

func (op Size) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('b')
	return hm.NewFnType(a, b)
}

func (op Size) ShapeExpr() shapes.Expr {
	return shapes.MakeArrow(
		shapes.Var('a'),
		shapes.IndexOf{I: shapes.Size(op), A: shapes.Var('a')},
	)
}

func (op Size) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	v := vs[0].(tensor.Tensor)
	shp := v.Shape()
	if shp.Eq(shapes.ScalarShape()) {
		return values.MakeScalarOf(v.Dtype(), 1), nil
	}

	return values.MakeScalarOf(v.Dtype(), shp[int(op)]), nil
}

func (op Size) String() string { return "Sz" }

// Repeat1 is the old style repeat
type Repeat1 struct {
	along int
}

func (op Repeat1) Arity() int             { return 1 }
func (op Repeat1) Type() hm.Type          { panic("NYI") }
func (op Repeat1) ShapeExpr() shapes.Expr { panic("NYI") }
func (op Repeat1) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	panic("NYI")
}
func (op Repeat1) String() string { return fmt.Sprintf("Repeat1_%d", op.along) }
func (op Repeat1) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	panic("NYI")
}

// Repeat :: along:int → n:int → a:Tensor → result:Tensor
type Repeat struct{}

func (op Repeat) Arity() int             { return 3 }
func (op Repeat) Type() hm.Type          { panic("NYI") }
func (op Repeat) ShapeExpr() shapes.Expr { panic("NYI") }
func (op Repeat) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	panic("NYI")
}
func (op Repeat) String() string { return fmt.Sprintf("Repeat%d_%d", op.n, op.along) }
func (op Repeat) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	panic("NYI")
}

type Slice struct{}

type sliceDiff struct{ Slice }

type Transpose struct{}

type Concat struct{}

type Reshape struct{}
