package stdops

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/exprgraph"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// MatMul is an op representing a matrix multiplication operation.
type MatMul[DT tensor.Num, T values.Value[DT]] struct{ binop }

// Type informs the type of the MatMul: Matrix a → Vector a → Vector a
func (op MatMul[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	return types.NewFunc(t, t, t)
}

// ShapeExpr informs the shape operations of MatMul: (a, b) → (b, c) → (a, c)
func (op MatMul[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	c := shapes.Var('c')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b, c},
		shapes.Abstract{a, c},
	)
}

func (op MatMul[DT, T]) do(ctx context.Context, a, b, prealloc T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())

	e := tensor.GetEngine(a, b)

	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	expShape := elimInnermostOutermost(a.Shape(), b.Shape())

	var ret tensor.Basic[DT]
	if ret, _, err = prepper.HandleFuncOpts(a, expShape, tensor.WithReuse(prealloc)); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	retVal = ret.(T)

	var bla tensor.BLA[DT, T]
	if bla, ok = e.(tensor.BLA[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}

	if err = bla.MatMul(ctx2, a, b, retVal, nil); err != nil {
		return retVal, err
	}

	task.End()
	return retVal, err
}

// Do performs the matrix multiplication.
func (op MatMul[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	var prealloc T
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs the matrix multiplication with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op MatMul[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}

// String implements fmt.Stringer.
func (op MatMul[DT, T]) String() string { return "×" }

// SymDiff performs symbolic differentiation for `MatMul`.
func (op MatMul[DT, T]) SymDiff(g *exprgraph.Graph, inputs []*exprgraph.Node, output, grade *exprgraph.Node) (retVal []*exprgraph.Node, err error) {
	panic("NYI")
}

// DoDiff allows automatic differentiation for `MatMul`.
func (op MatMul[DT, T]) DoDiff(ctx context.Context, inputs []gorgonia.Tensor, output gorgonia.Tensor) (err error) {
	adv := exprgraph.T2B[DT](inputs[0]).(*dual.Dual[DT, T])
	bdv := exprgraph.T2B[DT](inputs[1]).(*dual.Dual[DT, T])
	cdv := exprgraph.T2B[DT](output).(*dual.Dual[DT, T])

	advd := adv.Deriv()
	bdvd := bdv.Deriv()

	// temporary transpose
	var advT, bdvT T
	if bdvT, err = bdv.V().(tensor.Operable[T]).T(); err != nil {
		return err
	}
	if advT, err = adv.V().(tensor.Operable[T]).T(); err != nil {
		return err
	}

	// dA = C×B'
	if _, err := op.PreallocDo(ctx, advd, cdv.Value(), bdvT); err != nil {
		return err
	}

	// dB = A'×C
	if _, err := op.PreallocDo(ctx, bdvd, advT, cdv.Value()); err != nil {
		return err
	}
	return nil
}

/*



 MAT-VEC MUL


*/

// MatVecMul is an op representing a matrix-vector multiplication operations.
type MatVecMul[DT tensor.Num, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op MatVecMul[DT, T]) String() string { return "×" }

// Type informs the type of the MatVecMul: Matrix a → Vector a → Vector a
func (op MatVecMul[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(2, a) // Matrix a
	v := types.MakeTensorType(1, a) // Vector a
	return types.NewFunc(t, v, v)
}

// ShapeExpr informs the shape operations of MatVecMul: (a, b) → (b, ) → (a, )
func (op MatVecMul[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	b := shapes.Var('b')
	return shapes.MakeArrow(
		shapes.Abstract{a, b},
		shapes.Abstract{b},
		shapes.Abstract{a},
	)
}

func (op MatVecMul[DT, T]) do(ctx context.Context, a, b, prealloc T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())

	e := tensor.GetEngine(a, b)

	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}
	expShape := elimInnermostOutermost(a.Shape(), b.Shape())

	var ret tensor.Basic[DT]
	if ret, _, err = prepper.HandleFuncOpts(a, expShape, tensor.WithReuse(prealloc)); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	retVal = ret.(T)

	var bla tensor.BLA[DT, T]
	if bla, ok = e.(tensor.BLA[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}

	if err = bla.MatVecMul(ctx2, a, b, retVal, nil); err != nil {
		return retVal, err
	}

	task.End()
	return retVal, err
}

// Do performs the matrix-vector multiplication.
func (op MatVecMul[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	var prealloc T
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs the matrix-vector multiplication with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op MatVecMul[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}

// Inner is a op representing vector dot product (inner product) operations.
type Inner[DT tensor.Num, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op Inner[DT, T]) String() string { return "·" }

// Type informs the type of the Inner: Vector a → Vector a → a
func (op Inner[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	v := types.MakeTensorType(1, a) // Vector a
	return types.NewFunc(v, v, a)
}

// ShapeExpr informs the shape operations of Inner: (a, ) → (a, ) → ()
func (op Inner[DT, T]) ShapeExpr() shapes.Expr {
	a := shapes.Var('a')
	return shapes.MakeArrow(
		shapes.Abstract{a},
		shapes.Abstract{a},
		shapes.ScalarShape(),
	)
}

func (op Inner[DT, T]) do(ctx context.Context, a, b T) (retVal DT, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a, b)
	var bla tensor.InnerProder[DT, T]
	var ok bool
	if bla, ok = e.(tensor.InnerProder[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}
	retVal, err = bla.Inner(ctx2, a, b)
	task.End()
	return
}

// Do performs the inner product operation.
func (op Inner[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	ret, err := op.do(ctx, a, b)
	retVal, _ = values.AnyToScalar(ret)
	return retVal, err
}

// PreallocDo performs the inner product operation with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op Inner[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	ret, err := op.do(ctx, a, b)
	err = prealloc.SetAt(ret, 0)
	retVal = prealloc
	return retVal, err
}

// Outer is an op that represents outer product operations.
// Note that this op is not the higher order "outer" that one may be familiar with
// from vector languages like APL.
type Outer[DT tensor.Num, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op Outer[DT, T]) String() string { return "⊗" }

// Type informs the type of Outer: Tensor-n a → Tensor-n a → Matrix a
func (op Outer[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := types.MakeTensorType(-1, a)
	m := types.MakeTensorType(2, a)
	return types.NewFunc(t, t, m)
}

// ShapeExpr informs the shape operations of Outer: a → b → (Π a, Π b).
func (op Outer[DT, T]) ShapeExpr() shapes.Expr {
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

func (op Outer[DT, T]) do(ctx context.Context, a, b, prealloc T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}
	ctx2, task := trace.NewTask(ctx, op.String())
	e := tensor.GetEngine(a, b)

	var prepper tensor.FuncOptHandler[DT]
	var ok bool
	if prepper, ok = e.(tensor.FuncOptHandler[DT]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, prepper, errors.ThisFn())
	}

	var ret tensor.Basic[DT]
	expShape := shapes.Shape{a.Shape().TotalSize(), b.Shape().TotalSize()}
	if ret, _, err = prepper.HandleFuncOpts(a, expShape, tensor.WithReuse(prealloc)); err != nil {
		return retVal, errors.Wrapf(err, errors.FailedFuncOpt, errors.ThisFn())
	}
	retVal = ret.(T)

	var bla tensor.BLA[DT, T]
	if bla, ok = e.(tensor.BLA[DT, T]); !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, bla, errors.ThisFn())
	}
	// var incr []DT
	// if fo.Incr {
	// 	incr = make([]DT, len(retVal.Data()))
	// 	copy(incr, retVal.Data())
	// }

	if err = bla.Outer(ctx2, a, b, retVal, nil); err != nil {
		return retVal, err
	}
	task.End()
	return retVal, err
}

// Do performs the outer product operation.
func (op Outer[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	var prealloc T
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs the outer product operation with a preallocated value.
// PreallocDo allows MatMul to implement ops.PreallocDo
func (op Outer[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)

}
