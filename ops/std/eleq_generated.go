package stdops

// Code generated by genops, which is a ops generation tool for Gorgonia. DO NOT EDIT.

import (
	"context"
	"runtime/trace"

	"github.com/chewxy/hm"
	gctx "gorgonia.org/gorgonia/internal/context"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// elEqOp is the base op for elementwise equal-to.
type elEqOp[DT any, T values.Value[DT]] struct {
	binop
	retSame bool
}

// String implements fmt.Stringer.
func (op elEqOp[DT, T]) String() string { return "=" }

// Do performs elementwise equal-to.
func (op elEqOp[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	// Do the actual operation
	ctx2, task := trace.NewTask(ctx, op.String())
	if op.retSame {
		retVal, err = tensor.ElEq(a, b, tensor.WithContext(ctx2), tensor.AsSameType())
	} else {
		retVal, err = tensor.ElEq(a, b, tensor.WithContext(ctx2))
	}
	task.End()
	return retVal, err
}

// PreallocDo performs elementwise equal-to but with a preallocated return value.
// PreallocDo allows elEq to implement ops.PreallocOp.
func (op elEqOp[DT, T]) PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error) {
	if err := gctx.Handle(ctx); err != nil {
		return retVal, err
	}

	a := vs[0]
	b := vs[1]

	ctx2, task := trace.NewTask(ctx, op.String())
	if op.retSame {
		retVal, err = tensor.ElEq(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2), tensor.AsSameType())
	} else {
		retVal, err = tensor.ElEq(a, b, tensor.WithReuse(prealloc), tensor.WithContext(ctx2))
	}
	task.End()
	return retVal, err
}                                                  // DiffWRT returns {false, false} for elEq
func (op elEqOp[DT, T]) DiffWRT(inputs int) []bool { return twofalses }

// elEqVV is a tensor-tensor elementwise equal-to.
type elEqVV[DT any, T values.Value[DT]] struct {
	elEqOp[DT, T]
	binopVV
}

// Type returns the type: (·) : a → a → a or (·) :  a → a → b
func (op elEqVV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, a, a)
	}
	b := types.MakeDependent(a, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a, a, b)
}

// elEqVS is a tensor-scalar elementwise equal-to.
type elEqVS[DT any, T values.Value[DT]] struct {
	elEqOp[DT, T]
	binopVS
}

// String implements fmt.Stringer.
func (op elEqVS[DT, T]) String() string { return "=·" }

// Type returns the type: (·) : a → b → a or (·) :  a → b → c
func (op elEqVS) Type() hm.Type {
	a := hm.TypeVariable('a') // (T U) or U
	b := hm.TypeVariable('b') // U
	if op.retSame {
		return types.NewFunc(a, b, a)
	}
	c := types.MakeDependent(a, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}

// elEqSV is a scalar-tensor elementwise equal-to.
type elEqSV[DT any, T values.Value[DT]] struct {
	elEqOp[DT, T]
	binopSV
}

// String implements fmt.Stringer.
func (op elEqSV[DT, T]) String() string { return "·=" }

// Type returns the type: (·) : a → b → b or (·) :  a → b → c
func (op elEqSV[DT, T]) Type() hm.Type {
	a := hm.TypeVariable('a') // U
	b := hm.TypeVariable('b') // (T U) or U
	if op.retSame {
		return types.NewFunc(a, b, b)
	}
	c := types.MakeDependent(b, tensor.Bool) // (T Bool) or Bool
	return types.NewFunc(a, b, c)
}
