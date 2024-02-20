package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/dtype"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// ltOp is the base op for elementwise less-than.
type ltOp[DT any, T values.Value[DT], U values.Value[bool]] struct{ binop }

type ltOpRS[DT any, T values.Value[DT]] struct{ binop }

// String implements fmt.Stringer.
func (op ltOp[DT, T, U]) String() string { return "<" }

// String implements fmt.Stringer.
func (op ltOpRS[DT, T]) String() string { return "<" }

func (op ltOp[DT, T, U]) do(ctx context.Context, a, b T, prealloc U) (retVal U, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBinOpTrans[DT](a, b, tensor.WithReuse(prealloc))
	if err != nil {
		return retVal, err
	}
	e = e.BasicEng()

	asSame := fo.AsType == a.Dtype()
	toBroadcast := fo.Broadcast

	ord, ok := e.(tensor.Ord[DT, tensor.Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, ord, errors.ThisFn())
	}
	if fo.Incr {
		return retVal, errors.Errorf("Unable to perform Incr for lt")
	}
	switch {
	case toBroadcast:
		err = ord.LtBroadcastable(ctx, a, b, ret, asSame, newAPA, newAPB)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = ord.Lt(ctx2, a, b, ret, asSame)
	}
	retVal = ret.(U)
	return retVal, err
}

func (op ltOpRS[DT, T]) do(ctx context.Context, a, b, prealloc T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBinOpTrans[DT](a, b, tensor.WithReuse(prealloc), tensor.As(dtype.Datatype[DT]{}))
	if err != nil {
		return retVal, err
	}
	e = e.BasicEng()

	toBroadcast := fo.Broadcast

	ord, ok := e.(tensor.Ord[DT, tensor.Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, ord, errors.ThisFn())
	}
	if fo.Incr {
		return retVal, errors.Errorf("Unable to perform Incr for lt")
	}
	switch {
	case toBroadcast:
		err = ord.LtBroadcastable(ctx, a, b, ret, true, newAPA, newAPB)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = ord.Lt(ctx2, a, b, ret, true)
	}
	retVal = ret.(T)
	return retVal, err
}

// Do performs elementwise less-than.
func (op ltOp[DT, T, U]) Do(ctx context.Context, vs ...T) (retVal U, err error) {
	a := vs[0]
	b := vs[1]
	var prealloc U
	return op.do(ctx, a, b, prealloc)
}

// Do performs elementwise less-than.
func (op ltOpRS[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	var prealloc T
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs elementwise less-than but with a preallocated return value.
// PreallocDo allows lt to implement ops.PreallocOp.
func (op ltOp[DT, T, U]) PreallocDo(ctx context.Context, prealloc U, vs ...T) (retVal U, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs elementwise less-than but with a preallocated return value.
// PreallocDo allows lt to implement ops.PreallocOp.
func (op ltOpRS[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}                                                   // DiffWRT returns {false, false} for lt
func (op ltOp[DT, T, U]) DiffWRT(inputs int) []bool { return twofalses }

// DiffWRT returns {false, false} for lt
func (op ltOpRS[DT, T]) DiffWRT(inputs int) []bool { return twofalses }

// ltVV is a tensor-tensor elementwise less-than.
type ltVV[DT any, T values.Value[DT], U values.Value[bool]] struct {
	ltOp[DT, T, U]
	binopVV
}

type ltVVRS[DT any, T values.Value[DT]] struct {
	ltOpRS[DT, T]
	binopVV
}

// Type returns the type: (·) (·) :  a → a → b
func (op ltVV[DT, T, U]) Type() hm.Type {
	a := hm.TypeVariable('a')               // (T a) or a
	b := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, a, b)
}

// Type returns the type: (·) :  a → a → a
func (op ltVVRS[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T a) or a
	return types.NewFunc(a, a, a)
}

// ltVS is a tensor-scalar elementwise less-than.
type ltVS[DT any, T values.Value[DT], U values.Value[bool]] struct {
	ltOp[DT, T, U]
	binopVS
}

// ltVSRS is a tensor-scalar elementwise less-than.
type ltVSRS[DT any, T values.Value[DT]] struct {
	ltOpRS[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op ltVS[DT, T, U]) String() string { return "<·" }

// String implements fmt.Stringer.
func (op ltVSRS[DT, T]) String() string { return "<·" }

// Type returns the type: (·) :  a → b → c
func (op ltVS[DT, T, U]) Type() hm.Type {
	a := hm.TypeVariable('a')               // (T a)
	b := hm.TypeVariable('b')               // a
	c := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}

// Type returns the type: (·) : a → b → a
func (op ltVSRS[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T a) or a
	b := hm.TypeVariable('b') // b
	return types.NewFunc(a, b, a)
}

// ltSV is a scalar-tensor elementwise less-than.
type ltSV[DT any, T values.Value[DT], U values.Value[bool]] struct {
	ltOp[DT, T, U]
	binopSV
}

// ltSV is a scalar-tensor elementwise less-than.
type ltSVRS[DT any, T values.Value[DT]] struct {
	ltOpRS[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op ltSV[DT, T, U]) String() string { return "·<" }

// String implements fmt.Stringer.
func (op ltSVRS[DT, T]) String() string { return "·<" }

// Type returns the type: (·) :  a → b → c
func (op ltSV[DT, T, U]) Type() hm.Type {
	a := hm.TypeVariable('a')               // U
	b := hm.TypeVariable('b')               // (T U) or U
	c := types.MakeDependent(b, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}

// Type returns the type: (·) : a → b → b
func (op ltSVRS[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // a
	b := hm.TypeVariable('b') // (T b) or b
	return types.NewFunc(a, b, b)
}
