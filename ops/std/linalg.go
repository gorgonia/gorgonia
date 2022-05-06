package stdops

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/exprgraph"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// MatMul is an op representing a matrix multiplication operation.
type MatMul struct{ binop }

// Type informs the type of the MatMul: Matrix a → Vector a → Vector a
func (op MatMul) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	return types.NewFunc(t, t, t)
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
	if err := gctx.Handle(ctx); err != nil {
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
	if err := gctx.Handle(ctx); err != nil {
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

// SymDiff performs symbolic differentiation for `MatMul`.
func (op MatMul) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output, grade *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	panic("NYI")
}

// DoDiff allows automatic differentiation for `MatMul`.
func (op MatMul) DoDiff(ctx context.Context, inputs []Tensor, output Tensor) (err error) {
	adv := exprgraph.T2T(inputs[0]).(*dual.Dual)
	bdv := exprgraph.T2T(inputs[1]).(*dual.Dual)
	cdv := exprgraph.T2T(output).(*dual.Dual)

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	// temporary transpose
	if err := bdv.Value.T(); err != nil {
		return err
	}
	if err := adv.Value.T(); err != nil {
		return err
	}
	defer bdv.Value.UT()
	defer adv.Value.UT()

	// dA = C×B'
	if _, err := op.PreallocDo(ctx, advd, cdv.Value, bdv.Value); err != nil {
		return err
	}

	// dB = A'×C
	if _, err := op.PreallocDo(ctx, bdvd, adv.Value, cdv.Value); err != nil {
		return err
	}
	return nil
}

/*



 MAT-VEC MUL


*/

// MatVecMul is an op representing a matrix-vector multiplication operations.
type MatVecMul struct{ binop }

// String implements fmt.Stringer.
func (op MatVecMul) String() string { return "×" }

// Type informs the type of the MatVecMul: Matrix a → Vector a → Vector a
func (op MatVecMul) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	v := types.MakeTensorType(1, a) // Vector a
	return types.NewFunc(t, v, v)
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
	if err := gctx.Handle(ctx); err != nil {
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
	if err := gctx.Handle(ctx); err != nil {
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

// Type informs the type of the VecDot: Vector a → Vector a → a
func (op VecDot) Type() hm.Type {
	a := hm.TypeVariable('a')
	v := types.MakeTensorType(1, a) // Vector a
	return types.NewFunc(v, v, a)
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

// Do performs the inner product operation.
func (op VecDot) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := gctx.Handle(ctx); err != nil {
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

// PreallocDo performs the inner product operation with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op VecDot) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	ret, err := tensor.Inner(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	if err != nil {
		return nil, errors.Wrap(err, "VecDot.PreallocDo failed")
	}
	err = prealloc.SetAt(ret, 0)
	retVal := prealloc
	task.End()
	return retVal, err
}

// Outer is an op that represents outer product operations.
// Note that this op is not the higher order "outer" that one may be familiar with
// from vector languages like APL.
type Outer struct{ binop }

// String implements fmt.Stringer.
func (op Outer) String() string { return "⊗" }

// Type informs the type of Outer: Tensor-n a → Tensor-n a → Matrix a
func (op Outer) Type() hm.Type {

	a := hm.TypeVariable('a')
	t := types.MakeTensorType(-1, a)
	m := types.MakeTensorType(2, a)
	return types.NewFunc(t, t, m)
}

// ShapeExpr informs the shape operations of Outer: a → b → (Π a, Π b).
func (op Outer) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	return shapes.MakeArrow(
		a,
		b,
		shapes.Abstract{
			shapes.UnaryOp{shapes.Prod, a},
			shapes.UnaryOp{shapes.Prod, b},
		},
	)
}

// Do performs the outer product operation.
func (op Outer) Do(ctx context.Context, vs ...values.Value) (values.Value, error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}

	a := vs[0]
	b := vs[1]
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.Outer(a, b, tensor.WithContext(ctx2))

	task.End()

	return retVal, err
}

// PreallocDo performs the outer product operation with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op Outer) PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (values.Value, error) {
	if err := gctx.Handle(ctx); err != nil {
		return nil, err
	}
	a := vs[0].(tensor.Tensor)
	b := vs[1].(tensor.Tensor)
	ctx2, task := trace.NewTask(ctx, op.String())
	retVal, err := tensor.Outer(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	task.End()
	return retVal, err
}
