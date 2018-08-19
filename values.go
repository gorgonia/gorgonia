package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Value represents a value that Gorgonia accepts. At this point it is implemented by:
//		- all scalar value types (F64, F32... etc)
// 		- *tensor.Dense
// 		- *dualValue
//
// A Value is essentially any thing that knows its own type and shape.
// Most importantly though, a Value is a pointer - and can be converted into a tensor.Memory.
// This is done for the sake of interoperability with external devices like cgo or CUDA or OpenCL.
// This also means for the most part most Values will be allocated on the heap.
// There are some performance tradeoffs made in this decision, but ultimately this is better than having to manually manage blocks of memory
type Value interface {
	Shape() tensor.Shape // Shape  returns the shape of the Value. Scalar values return ScalarShape()
	Size() int           // Size represents the number of elements in the Value. Note that in cases such as a *tensor.Dense, the underlying slice MAY have more elements than the Size() reports. This is correct.
	Data() interface{}   // Data returns the original representation of the Value
	Dtype() tensor.Dtype // Dtype returns the Dtype of the value

	tensor.Memory
	fmt.Formatter
}

// Valuer is any type that can return a Value
type Valuer interface {
	Value() Value
}

// Zeroer is a Value that can zero itself
type Zeroer interface {
	Value
	Zero()
}

// ZeroValuer is a a Value that can provide the zero-value of its type
type ZeroValuer interface {
	Value
	ZeroValue() Value
}

// Dtyper represents any type (typically a Value) that knows its own Dtype
type Dtyper interface {
	Dtype() tensor.Dtype
}

// Typer represents any type (typically a Op) that knows its own Type
type Typer interface {
	Type() hm.Type
}

// ValueEqualer represents any type that can perform a equal value check
type ValueEqualer interface {
	ValueEq(Value) bool
}

// ValueCloser represents any type that can perform a close-value check
type ValueCloser interface {
	ValueClose(interface{}) bool
}

// Cloner represents any type that can clone itself.
type Cloner interface {
	Clone() interface{}
}

// CloneErrorer represents any type that can clone itself and return an error if necessary
type CloneErrorer interface {
	Clone() (interface{}, error)
}

// CopierTo represents any type that can copy data to the destination.
type CopierTo interface {
	CopyTo(dest interface{}) error
}

// CopierFrom represents any type that can copy data from the source provided.
type CopierFrom interface {
	CopyFrom(src interface{}) error
}

// Setter is a any value that can Memset itself to the provided value
// type Setter interface {
// 	SetAll(interface{}) error
// }

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
