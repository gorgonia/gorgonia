package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// makeValue creates a value given a type and shape. The default value is the zero value of the type.
func makeValue(t hm.Type, s tensor.Shape) (retVal Value, err error) {
	var dt tensor.Dtype
	if dt, err = dtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		switch dt {
		case tensor.Float64:
			return newF64(0), nil
		case tensor.Float32:
			return newF32(0), nil
		case tensor.Int:
			return newI(0), nil
		case tensor.Int64:
			return newI64(0), nil
		case tensor.Int32:
			return newI32(0), nil
		case tensor.Byte:
			return newU8(0), nil
		case tensor.Bool:
			return newB(false), nil
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

func makeValueFromMem(t hm.Type, s tensor.Shape, mem tensor.Memory) (retVal Value, err error) {
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

func makeScalarFromMem(dt tensor.Dtype, mem tensor.Memory) (retVal Value, err error) {
	switch dt {
	case tensor.Float64:
		retVal = (*F64)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Float32:
		retVal = (*F32)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int:
		retVal = (*I)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int64:
		retVal = (*I64)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Int32:
		retVal = (*I32)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Byte:
		retVal = (*U8)(unsafe.Pointer(mem.Uintptr()))
	case tensor.Bool:
		retVal = (*B)(unsafe.Pointer(mem.Uintptr()))
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
// The Value passed in are either Scalar, tensor.Tensor, or *dualValue. Anything else will panic.
func ScalarAsTensor(v Value, dims int, e tensor.Engine) Value {
	switch a := v.(type) {
	case Scalar:
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
