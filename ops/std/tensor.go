package stdops

import (
	"context"
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
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
type Size uint

func (op Size) Arity() int { return 1 }

func (op Size) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(0, types.Ptr) // types.Ptr is a dummy dtype. types.Dependent relies on the dtype of `a` and the dims of `t`
	return hm.NewFnType(a, types.MakeDependent(t, a))
}

func (op Size) ShapeExpr() shapes.Expr {
	return shapes.MakeArrow(
		shapes.Var('a'),
		// shapes.IndexOf{I: shapes.Size(op), A: shapes.Var('a')},
		shapes.ScalarShape(),
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
	if int(op) >= shp.Dims() {
		return nil, errors.Errorf("Input has shape %v, which has %d dims. Want dim %d.", shp, shp.Dims(), int(op))
	}

	return values.MakeScalarOf(v.Dtype(), shp[int(op)]), nil
}

func (op Size) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	v := vs[0].(tensor.Tensor)
	shp := v.Shape()
	if shp.Eq(shapes.ScalarShape()) {
		err = values.CopyScalarOf(v.Dtype(), prealloc.(values.Scalar), 1)
		return prealloc, err
	}
	if int(op) >= shp.Dims() {
		return nil, errors.Errorf("Input has shape %v, which has %d dims. Want dim %d.", shp, shp.Dims(), int(op))
	}
	err = values.CopyScalarOf(v.Dtype(), prealloc.(values.Scalar), shp[int(op)])
	return prealloc, err

}

func (op Size) String() string { return fmt.Sprintf("Sz[%d]", int(op)) }

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
type Repeat struct {
	n     int
	along int
}

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

// Slice _{Slices} :: Tensor → a
// The list of slices are parameterized
type Slice struct {
	Slices shapes.Slices
}

func (op Slice) Arity() int { return 1 }
func (op Slice) Type() hm.Type {
	a := hm.TypeVariable('a')
	r := &types.Sliced{Of: a, Along: op.Slices}
	return hm.NewFnType(a, r)
}

func (op Slice) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	r := shapes.SliceOf{
		Slice: op.Slices,
		A:     a,
	}
	return shapes.MakeArrow(a, r)
}

func (op Slice) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	v := vs[0]
	return v.Slice(op.Slices...)
}

func (op Slice) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	v := vs[0]
	if s, ok := v.(tensor.SlicerInto); ok {
		return s.SliceInto(prealloc.(tensor.Tensor), op.Slices...)
	}
	return nil, errors.Errorf("NYI: preallocdo")
}

func (op Slice) String() string { return fmt.Sprintf("%v", op.Slices) }

type sliceDiff struct{ Slice }

type Concat struct{}

// Reshape is an Op representing a reshape operation.
type Reshape struct {
	To shapes.Shape
}

func (op *Reshape) Arity() int { return 1 }
func (op *Reshape) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(op.To.Dims(), types.Ptr) // here we use types.Ptr but because we are using types.Dependent, the resulting type only cares about the dims of `t`, not the `t.Of`
	return hm.NewFnType(a, types.MakeDependent(t, a))
}
func (op *Reshape) ShapeExpr() shapes.Expr { return shapes.MakeArrow(shapes.Var('a'), op.To) } // TODO: take advantage of shapes library's checking options

func (op *Reshape) Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	v, err := values.ShallowClone(vs[0])
	if err != nil {
		return nil, errors.Wrapf(err, "Reshape failed. Cannot reshape %v to %v", v.Shape(), op.To)
	}

	if err = v.Reshape(op.To...); err != nil {
		return nil, errors.Wrapf(err, "Reshape failed. Cannot reshape %v to %v", v.Shape(), op.To)
	}
	return v, nil

}

func (op *Reshape) String() string { return fmt.Sprintf("ReshapeTo %v", op.To) }

func (op *Reshape) DiffWRT(inputs int) []bool { return onetrue }
