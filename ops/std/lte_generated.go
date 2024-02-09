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

// lteOp is the base op for elementwise less-than-or-equal-to.
type lteOp[DT any, T values.Value[DT]] struct {
	binop
	retSame bool
}

// String implements fmt.Stringer.
func (op lteOp[DT, T]) String() string { return "≤" }

func (op lteOp[DT, T]) do(ctx context.Context, a, b, prealloc T) (retVal T, err error) {
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

	asSame := fo.AsType == a.Dtype()
	toBroadcast := fo.Broadcast

	ord, ok := e.(tensor.Ord[DT, tensor.Basic[DT]])
	if !ok {
		return retVal, errors.Errorf(errors.EngineSupport, e, ord, errors.ThisFn())
	}
	if fo.Incr {
		return retVal, errors.Errorf("Unable to perform Incr for lte")
	}
	switch {
	case toBroadcast:
		err = ord.LteBroadcastable(ctx, a, b, ret, asSame, newAPA, newAPB)
	default:
		if err := checkCompatibleShape(a.Shape(), b.Shape()); err != nil {
			return retVal, err
		}
		err = ord.Lte(ctx2, a, b, ret, asSame)
	}
	retVal = ret.(T)
	return retVal, err
}

// Do performs elementwise less-than-or-equal-to.
func (op lteOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	var prealloc T
	return op.do(ctx, a, b, prealloc)
}

// PreallocDo performs elementwise less-than-or-equal-to but with a preallocated return value.
// PreallocDo allows lte to implement ops.PreallocOp.
func (op lteOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	a := vs[0]
	b := vs[1]
	return op.do(ctx, a, b, prealloc)
}                                                 // DiffWRT returns {false, false} for lte
func (op lteOp[DT, T]) DiffWRT(inputs int) []bool { return twofalses }

// lteVV is a tensor-tensor elementwise less-than-or-equal-to.
type lteVV[DT any, T values.Value[DT]] struct {
	lteOp[DT, T]
	binopVV
}

// Type returns the type: (·) : a → a → a or (·) :  a → a → b
func (op lteVV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, a, a)
	}
	b := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, a, b)
}

// lteVS is a tensor-scalar elementwise less-than-or-equal-to.
type lteVS[DT any, T values.Value[DT]] struct {
	lteOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op lteVS[DT, T]) String() string { return "≤·" }

// Type returns the type: (·) : a → b → a or (·) :  a → b → c
func (op lteVS[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	b := hm.TypeVariable('b') // U
	if op.retSame {
		return types.NewFunc(a, b, a)
	}
	c := types.MakeDependent(a, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}

// lteSV is a scalar-tensor elementwise less-than-or-equal-to.
type lteSV[DT any, T values.Value[DT]] struct {
	lteOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op lteSV[DT, T]) String() string { return "·≤" }

// Type returns the type: (·) : a → b → b or (·) :  a → b → c
func (op lteSV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // U
	b := hm.TypeVariable('b') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, b, b)
	}
	c := types.MakeDependent(b, dtype.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}
