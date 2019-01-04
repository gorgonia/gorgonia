package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/primitive"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// makeValue creates a value given a type and shape. The default value is the zero value of the type.
func makeValue(t hm.Type, s tensor.Shape) (retVal value.Value, err error) {
	var dt tensor.Dtype
	if dt, err = dtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		switch dt {
		case tensor.Float64:
			return primitive.NewF64(0), nil
		case tensor.Float32:
			return primitive.NewF32(0), nil
		case tensor.Int:
			return primitive.NewI(0), nil
		case tensor.Int64:
			return primitive.NewI64(0), nil
		case tensor.Int32:
			return primitive.NewI32(0), nil
		case tensor.Byte:
			return primitive.NewU8(0), nil
		case tensor.Bool:
			return primitive.NewB(false), nil
		}
	}

	switch tt := t.(type) {
	case TensorType:
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...)), nil
	default:
		err = errors.Errorf(nyiTypeFail, "MakeValue", tt)
		return
	}
}

func makeValueFromMem(t hm.Type, s tensor.Shape, mem tensor.Memory) (retVal value.Value, err error) {
	var dt tensor.Dtype
	if dt, err = dtypeOf(t); err != nil {
		return
	}
	if s.IsScalar() {
		return makeScalarFromMem(dt, mem)
	}

	switch tt := t.(type) {
	case TensorType:
		memsize := calcMemSize(dt, s)
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...), tensor.FromMemory(mem.Uintptr(), uintptr(memsize))), nil
	case tensor.Dtype:
		return makeScalarFromMem(tt, mem)
	default:
		err = errors.Errorf(nyiTypeFail, "MakeValue", tt)
		return
	}
}

func makeScalarFromMem(dt tensor.Dtype, mem tensor.Memory) (retVal value.Value, err error) {
	switch dt {
	case tensor.Float64:
		retVal = (*primitive.F64)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Float32:
		retVal = (*primitive.F32)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int:
		retVal = (*primitive.I)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int64:
		retVal = (*primitive.I64)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int32:
		retVal = (*primitive.I32)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Byte:
		retVal = (*primitive.U8)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Bool:
		retVal = (*primitive.B)(unsafe.Pointer(mem.Uintptr()))
	default:
		err = errors.Errorf(nyiTypeFail, "makeScalarFromMem", dt)
	}
	return
}

func logicalSize(s tensor.Shape) int {
	if s.IsScalar() {
		return 1
	}
	return s.TotalSize()
}

func calcMemSize(dt tensor.Dtype, s tensor.Shape) int64 {
	var elemSize int64
	if s.IsScalar() {
		elemSize = 1
	} else {
		elemSize = int64(s.TotalSize())
	}
	dtSize := int64(dt.Size())
	return elemSize * dtSize
}

// ScalarAsTensor returns the tensor representation of a scalar. It is particularly useful as a "reshape" of tensors of sorts
//
// The value.Value passed in are either value.Scalar, tensor.Tensor, or *dualValue. Anything else will panic.
func ScalarAsTensor(v value.Value, dims int, e tensor.Engine) value.Value {
	switch a := v.(type) {
	case value.Scalar:
		sh := make(tensor.Shape, dims)
		for i := range sh {
			sh[i] = 1
		}
		return tensor.New(tensor.WithShape(sh...), tensor.Of(a.Dtype()), tensor.FromMemory(a.Uintptr(), a.MemSize()), tensor.WithEngine(e))
	case tensor.Tensor:
		return a
	case *dualValue:
		b := new(dualValue)
		b.Value = ScalarAsTensor(a.Value, dims, e)
		b.d = ScalarAsTensor(a.d, dims, e)
		return b
	case nil:
		return nil
	default:
		panic(fmt.Sprintf("Unable to convert %v to Tensor", v))
	}
}
