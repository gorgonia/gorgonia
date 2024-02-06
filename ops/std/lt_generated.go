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
type ltOp[DT any, T values.Value[DT]] struct {
	binop
	retSame bool
}

// String implements fmt.Stringer.
func (op ltOp[DT, T]) String() string { return "<" }

// Do performs elementwise less-than.
func (op ltOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBinOpTrans[DT](a, b)
	if err != nil {
		return retVal, err
	}

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
	retVal = ret.(T)
	return retVal, err
}

// PreallocDo performs elementwise less-than but with a preallocated return value.
// PreallocDo allows lt to implement ops.PreallocOp.
func (op ltOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	defer task.End()

	e, newAPA, newAPB, ret, fo, err := tensor.PrepBinOpTrans[DT](a, b, tensor.WithReuse(prealloc))
	if err != nil {
		return retVal, err
	}

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
	retVal = ret.(T)
	return retVal, err
}                                                // DiffWRT returns {false, false} for lt
func (op ltOp[DT, T]) DiffWRT(inputs int) []bool { return twofalses }

// ltVV is a tensor-tensor elementwise less-than.
type ltVV[DT any, T values.Value[DT]] struct {
	ltOp[DT, T]
	binopVV
}

// Type returns the type: (·) : a → a → a or (·) :  a → a → b
func (op ltVV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, a, a)
	}
	b := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, a, b)
}

// ltVS is a tensor-scalar elementwise less-than.
type ltVS[DT any, T values.Value[DT]] struct {
	ltOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op ltVS[DT, T]) String() string { return "<·" }

// Type returns the type: (·) : a → b → a or (·) :  a → b → c
func (op ltVS[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	b := hm.TypeVariable('b') // U
	if op.retSame {
		return types.NewFunc(a, b, a)
	}
	c := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}

// ltSV is a scalar-tensor elementwise less-than.
type ltSV[DT any, T values.Value[DT]] struct {
	ltOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op ltSV[DT, T]) String() string { return "·<" }

// Type returns the type: (·) : a → b → b or (·) :  a → b → c
func (op ltSV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // U
	b := hm.TypeVariable('b') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, b, b)
	}
	c := types.MakeDependent(b, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}
