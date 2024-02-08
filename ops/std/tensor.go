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
	"gorgonia.org/tensor/dense"
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
type At[DT any, T values.Value[DT]] []int

func (op At[DT, T]) Arity() int { return 1 }

func (op At[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.TensorType{Dims: len(op), Of: a}
	return hm.NewFnType(t, a)
}

func (op At[DT, T]) ShapeExpr() shapes.Expr {
	return shapes.MakeArrow(shapes.Var('a'), shapes.ScalarShape())
} // TODO: leverage shape package to actually add more checks

func (op At[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	v := any(vs[0]).(tensor.ValueGetter[DT])
	_, task := trace.NewTask(ctx, op.String())
	var r DT

	if r, err = v.At(op...); err == nil {
		ret, _ := values.AnyToScalar(r)
		retVal = ret.(T)
	}
	task.End()
	return retVal, err

}

func (op At[DT, T]) String() string { return fmt.Sprintf("At(%v)", []int(op)) }

// Size is an operation that gets the size of a given axis in a shape. It returns the size in the datatype of the input.
//
// Size is a HKOp[DT, Size, T, Size].
type Size[DT any, T values.Value[DT]] uint

func (op Size[DT, T]) Arity() int { return 1 }

func (op Size[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(0, types.Ptr) // types.Ptr is a dummy dtype. types.Dependent relies on the dtype of `a` and the dims of `t`
	return hm.NewFnType(a, types.MakeDependent(t, a))
}

func (op Size[DT, T]) ShapeExpr() shapes.Expr {
	return shapes.MakeArrow(
		shapes.Var('a'),
		// shapes.IndexOf{I: shapes.Size(op), A: shapes.Var('a')},
		shapes.ScalarShape(),
	)
}

func (op Size[DT, T]) Do(ctx context.Context, vs ...T) (retVal values.Size, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return -1, err
	}
	v := vs[0]
	shp := v.Shape()
	if shp.Eq(shapes.ScalarShape()) {
		return values.Size(1), nil
	}
	if int(op) >= shp.Dims() {
		return -1, errors.Errorf("Input has shape %v, which has %d dims. Want dim %d.", shp, shp.Dims(), int(op))
	}
	return values.Size(shp[int(op)]), nil
}

func (op Size[DT, T]) String() string { return fmt.Sprintf("Sz[%d]", int(op)) }

// Repeat1 is the old style repeat
type Repeat1[DT any, T values.Value[DT]] struct {
	along int
}

func (op Repeat1[DT, T]) Arity() int             { return 1 }
func (op Repeat1[DT, T]) Type() hm.Type          { panic("NYI") }
func (op Repeat1[DT, T]) ShapeExpr() shapes.Expr { panic("NYI") }
func (op Repeat1[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	panic("NYI")
}
func (op Repeat1[DT, T]) String() string { return fmt.Sprintf("Repeat1_%d", op.along) }
func (op Repeat1[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	panic("NYI")
}

// Repeat :: along:int → n:int → a:Tensor → result:Tensor
type Repeat[DT any, T values.Value[DT]] struct {
	n     int
	along int
}

func (op Repeat[DT, T]) Arity() int             { return 3 }
func (op Repeat[DT, T]) Type() hm.Type          { panic("NYI") }
func (op Repeat[DT, T]) ShapeExpr() shapes.Expr { panic("NYI") }
func (op Repeat[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	panic("NYI")
}
func (op Repeat[DT, T]) String() string { return fmt.Sprintf("Repeat%d_%d", op.n, op.along) }
func (op Repeat[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	panic("NYI")
}

// Slice _{Slices} :: Tensor → a
// The list of slices are parameterized
type Slice[DT any, T values.Value[DT]] struct {
	Slices shapes.Slices
}

func (op Slice[DT, T]) Arity() int { return 1 }
func (op Slice[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	r := &types.Sliced{Of: a, Along: op.Slices}
	return hm.NewFnType(a, r)
}

func (op Slice[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	r := shapes.SliceOf{
		Slice: op.Slices,
		A:     a,
	}
	return shapes.MakeArrow(a, r)
}

func (op Slice[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	v := any(vs[0]).(tensor.Operable[T])
	return v.Slice(op.Slices...)
}

func (op Slice[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	v := vs[0]
	if s, ok := any(v).(tensor.SlicerInto); ok {
		return s.SliceInto(prealloc, op.Slices...)
	}
	return retVal, errors.Errorf("NYI: preallocdo")
}

func (op Slice[DT, T]) String() string            { return fmt.Sprintf("%v", op.Slices) }
func (op Slice[DT, T]) DiffWRT(inputs int) []bool { return onetrue }

type sliceDiff[DT any, T values.Value[DT]] struct{ Slice[DT, T] }

func (op sliceDiff[DT, T]) Arity() int { return 2 }
func (op sliceDiff[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('b')
	return hm.NewFnType(a, b, a)
}
func (op sliceDiff[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	r := shapes.SliceOf{
		Slice: op.Slices,
		A:     a,
	}
	return shapes.MakeArrow(a, r, a)
}

func (op sliceDiff[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	t := vs[0]

	switch t := any(t).(type) {
	case *dense.Dense[DT]:
		grad := dense.New[DT](tensor.WithShape(t.Shape().Clone()...))
		outGrad := any(vs[1]).(*dense.Dense[DT])
		var v *dense.Dense[DT]
		if v, err = grad.Slice(op.Slices...); err != nil {
			return retVal, errors.Wrapf(err, "Failed to perform sliceDiff %v", op.Slices)
		}
		if _, err = v.Add(outGrad, tensor.UseUnsafe); err != nil {
			return retVal, err
		}
		retVal = any(grad).(T)
		return
	default:
		return retVal, errors.Errorf("NYI %T", t)
	}
}
func (op sliceDiff[DT, T]) String() string { return fmt.Sprintf("∂%v", op.Slices) }

type Concat[DT any, T values.Value[DT]] struct{}

// Reshape is an Op representing a reshape operation.
type Reshape[DT any, T values.Value[DT]] struct {
	To shapes.Shape
}

func (op *Reshape[DT, T]) Arity() int { return 1 }
func (op *Reshape[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(op.To.Dims(), types.Ptr) // here we use types.Ptr but because we are using types.Dependent, the resulting type only cares about the dims of `t`, not the `t.Of`
	return hm.NewFnType(a, types.MakeDependent(t, a))
}
func (op *Reshape[DT, T]) ShapeExpr() shapes.Expr { return shapes.MakeArrow(shapes.Var('a'), op.To) } // TODO: take advantage of shapes library's checking options

func (op *Reshape[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	v := values.ShallowClone(any(vs[0]).(tensor.ShallowCloner[T]))

	if err = v.Reshape(op.To...); err != nil {
		return retVal, errors.Wrapf(err, "Reshape failed. Cannot reshape %v to %v", v.Shape(), op.To)
	}
	return v, nil

}

func (op *Reshape[DT, T]) String() string { return fmt.Sprintf("ReshapeTo %v", op.To) }

func (op *Reshape[DT, T]) DiffWRT(inputs int) []bool { return onetrue }
