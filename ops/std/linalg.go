package stdops

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// MatMul is an op representing a matrix multiplication operation.
type MatMul struct{ binop }

// Type informs the type of the MatMul: Matrix a → Vector a → Vector a
func (op MatMul) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	return hm.NewFnType(t, t, t)
}

// ShapeExpr informs the shape operations of MatMul: (a, b) → (b, c) → (a, c)
func (op MatMul) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	c := shapes.Var('c')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b, c},
		shapes.Abstract{a, c},
	)
}

// Do performs the matrix multiplication.
func (op MatMul) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatMul(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs the matrix multiplication with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op MatMul) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatMul(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// String implements fmt.Stringer.
func (op MatMul) String() string { return "×" }

// MatVecMul is an op representing a matrix-vector multiplication operations.
type MatVecMul struct{ binop }

// String implements fmt.Stringer.
func (op MatVecMul) String() string { return "×" }

// Type informs the type of the MatVecMul: Matrix a → Vector a → Vector a
func (op MatVecMul) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	v := types.MakeTensorType(1, a) // Vector a
	return hm.NewFnType(t, v, v)
}

// ShapeExpr informs the shape operations of MatVecMul: (a, b) → (b, ) → (a, )
func (op MatVecMul) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b},
		shapes.Abstract{a},
	)
}

// Do performs the matrix-vector multiplication.
func (op MatVecMul) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatVecMul(a, b, tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// PreallocDo performs the matrix-vector multiplication with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op MatVecMul) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.MatVecMul(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}

// VecDot is a op representing vector dot product (inner product) operations.
type VecDot struct{ binop }

// String implements fmt.Stringer.
func (op VecDot) String() string { return "·" }

// Type informs the type of the MatMul: Vector a → Vector a → a
func (op VecDot) Type() hm.Type {
	a := hm.TypeVariable('a')
	v := types.MakeTensorType(1, a) // Vector a
	return hm.NewFnType(v, v, a)
}

// ShapeExpr informs the shape operations of VecDot: (a, ) → (a, ) → ()
func (op VecDot) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(
		shapes.Abstract{a},
		shapes.Abstract{a},
		shapes.ScalarShape(),
	)
}

// Do performs the matrix-vector multiplication.
func (op VecDot) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]
	ctx2, task := trace.NewTask(ctx, op.String())
	ret, err := tensor.Inner(a, b, tensor.WithContext(ctx2))
	retVal, _ := values.AnyToScalar(ret)
	task.End()

	return retVal, err
}

// PreallocDo performs the matrix-vector multiplication with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op VecDot) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := handleCtx(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	ret, err := tensor.Inner(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	retVal, _ := values.AnyToScalar(ret)
	task.End()
	return retVal, err
}
