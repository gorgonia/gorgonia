package value

import (
	"bytes"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

var (
	// Represents the types that Nodes can take in Gorgonia

	scalarShape = tensor.ScalarShape()
)

// F64 represents a float64 value.
type F64 float64

// F32 represents a float32 value.
type F32 float32

// I represents a int value.
type I int

// I64 represents a int64 value.
type I64 int64

// I32 represents a int32 value.
type I32 int32

// U8 represents a byte value.
type U8 byte

// B represents a bool value.
type B bool

// NewF64 ...
func NewF64(v float64) *F64 { r := F64(v); return &r }

// NewF32 ...
func NewF32(v float32) *F32 { r := F32(v); return &r }

// NewI ...
func NewI(v int) *I { r := I(v); return &r }

// NewI64 ...
func NewI64(v int64) *I64 { r := I64(v); return &r }

// NewI32 ...
func NewI32(v int32) *I32 { r := I32(v); return &r }

// NewU8 ...
func NewU8(v byte) *U8 { r := U8(v); return &r }

// NewB ...
func NewB(v bool) *B { r := B(v); return &r }

/* Shape() */

// Shape returns a scalar shape for all scalar values
func (v *F64) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *F32) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *I) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *I64) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *I32) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *U8) Shape() tensor.Shape { return scalarShape }

// Shape returns a scalar shape for all scalar values
func (v *B) Shape() tensor.Shape { return scalarShape }

// Size returns 0 for all scalar Values
func (v *F64) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *F32) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *I) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *I64) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *I32) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *U8) Size() int { return 0 }

// Size returns 0 for all scalar Values
func (v *B) Size() int { return 0 }

/* Data() */

// Data returns the original representation of the Value
func (v *F64) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *F32) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *I) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *I64) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *I32) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *U8) Data() interface{} { return v.Any() }

// Data returns the original representation of the Value
func (v *B) Data() interface{} { return v.Any() }

// Any ...
func (v *F64) Any() float64 { return float64(*v) }

// Any ...
func (v *F32) Any() float32 { return float32(*v) }

// Any ...
func (v *I) Any() int { return int(*v) }

// Any ...
func (v *I64) Any() int64 { return int64(*v) }

// Any ...
func (v *I32) Any() int32 { return int32(*v) }

// Any ...
func (v *U8) Any() byte { return byte(*v) }

// Any ...
func (v *B) Any() bool { return bool(*v) }

/* implements fmt.Formatter */

// Format implements fmt.Formatter
func (v *F64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *F32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *I) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *I64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *I32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *U8) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

// Format implements fmt.Formatter
func (v *B) Format(s fmt.State, c rune) { formatScalar(v, s, c) }

/* Dtype() */

// Dtype  returns the Dtype of the value
func (v *F64) Dtype() tensor.Dtype { return tensor.Float64 }

// Dtype  returns the Dtype of the value
func (v *F32) Dtype() tensor.Dtype { return tensor.Float32 }

// Dtype  returns the Dtype of the value
func (v *I) Dtype() tensor.Dtype { return tensor.Int }

// Dtype  returns the Dtype of the value
func (v *I64) Dtype() tensor.Dtype { return tensor.Int64 }

// Dtype  returns the Dtype of the value
func (v *I32) Dtype() tensor.Dtype { return tensor.Int32 }

// Dtype  returns the Dtype of the value
func (v *U8) Dtype() tensor.Dtype { return tensor.Byte }

// Dtype  returns the Dtype of the value
func (v *B) Dtype() tensor.Dtype { return tensor.Bool }

/* isScalar */

// IsScalar ...
func (v *F64) IsScalar() bool { return true }

// IsScalar ...
func (v *F32) IsScalar() bool { return true }

// IsScalar ...
func (v *I) IsScalar() bool { return true }

// IsScalar ...
func (v *I64) IsScalar() bool { return true }

// IsScalar ...
func (v *I32) IsScalar() bool { return true }

// IsScalar ...
func (v *U8) IsScalar() bool { return true }

// IsScalar ...
func (v *B) IsScalar() bool { return true }

/* Uintptr */

// Uintptr satisfies the tensor.Memory interface
func (v *F64) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *F32) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *I) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *I64) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *I32) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *U8) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

// Uintptr satisfies the tensor.Memory interface
func (v *B) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }

/* MemSize */

// MemSize satisfies the tensor.Memory interface
func (v *F64) MemSize() uintptr { return 8 }

// MemSize satisfies the tensor.Memory interface
func (v *F32) MemSize() uintptr { return 4 }

// MemSize satisfies the tensor.Memory interface
func (v *I) MemSize() uintptr { return reflect.TypeOf(*v).Size() }

// MemSize satisfies the tensor.Memory interface
func (v *I64) MemSize() uintptr { return 8 }

// MemSize satisfies the tensor.Memory interface
func (v *I32) MemSize() uintptr { return 4 }

// MemSize satisfies the tensor.Memory interface
func (v *U8) MemSize() uintptr { return 1 }

// MemSize satisfies the tensor.Memory interface
func (v *B) MemSize() uintptr { return reflect.TypeOf(*v).Size() }

/* Pointer */

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *F64) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *F32) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *I) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *I64) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *I32) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *U8) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

// Pointer returns the pointer as an unsafe.Pointer. Satisfies the tensor.Memory interface
func (v *B) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }

func formatScalar(v Scalar, s fmt.State, c rune) {
	var buf bytes.Buffer
	var ok bool

	buf.WriteRune('%')

	var width int
	if width, ok = s.Width(); ok {
		fmt.Fprintf(&buf, "%d", width)
	}

	var prec int
	if prec, ok = s.Precision(); ok {
		fmt.Fprintf(&buf, ".%d", prec)
	}

	switch c {
	case 's':
		buf.WriteRune('v')
	case 'd':
		switch v.(type) {
		case *F64, *F32, *U8, *B:
			buf.WriteRune('v')
		default:
			buf.WriteRune(c)
		}
	case 'f', 'g':
		switch v.(type) {
		case *I, *I64, *I32, *U8, *B:
			buf.WriteRune('v')
		default:
			buf.WriteRune(c)
		}
	default:
		buf.WriteRune(c)
	}

	if s.Flag('+') {
		s.Write([]byte(v.Dtype().String()))
		s.Write([]byte{' '})
	}

	fmt.Fprintf(s, buf.String(), v.Data())
}

// AnyToScalar turns any compatible value into a scalar
func AnyToScalar(any interface{}) (Scalar, tensor.Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, at.Dtype()
	case float64:
		return NewF64(at), tensor.Float64
	case float32:
		return NewF32(at), tensor.Float32
	case int:
		return NewI(at), tensor.Int
	case int32:
		return NewI32(at), tensor.Int32
	case int64:
		return NewI64(at), tensor.Int64
	case byte:
		return NewU8(at), tensor.Byte
	case bool:
		return NewB(at), tensor.Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled", any, any))
	}
}

// AnyToValue ...
func AnyToValue(any interface{}) (val Value, t hm.Type, dt tensor.Dtype, err error) {
	switch a := any.(type) {
	case Value:
		val = a
		t = TypeOf(a)
		dt = a.Dtype()
		return
	case float64, float32, int, int64, int32, byte, bool:
		val, dt = AnyToScalar(any)
		t = dt
		return
	case F64:
		return NewF64(float64(a)), tensor.Float64, tensor.Float64, nil
	case F32:
		return NewF32(float32(a)), tensor.Float32, tensor.Float32, nil
	case I:
		return NewI(int(a)), tensor.Int, tensor.Int, nil
	case I64:
		return NewI64(int64(a)), tensor.Int64, tensor.Int64, nil
	case I32:
		return NewI32(int32(a)), tensor.Int32, tensor.Int32, nil
	case U8:
		return NewU8(byte(a)), tensor.Uint8, tensor.Uint8, nil
	case B:
		return NewB(bool(a)), tensor.Bool, tensor.Bool, nil
	case tensor.Tensor:
		val = a
		t = TypeOf(a)
		dt = a.Dtype()
		return
	default:
		err = errors.Errorf("value %v of %T not yet handled", any, any)
		return
	}
}

// One ...
func One(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return NewF64(float64(1))
	case tensor.Float32:
		return NewF32(float32(1))
	case tensor.Int:
		return NewI(1)
	case tensor.Int32:
		return NewI32(int32(1))
	case tensor.Int64:
		return NewI64(int64(1))
	case tensor.Byte:
		return NewU8(byte(1))
	case tensor.Bool:
		return NewB(true)
	default:
		panic("Unhandled dtype")
	}
}

// Zero returns the zero value or the given type
func Zero(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return NewF64(float64(0))
	case tensor.Float32:
		return NewF32(float32(0))
	case tensor.Int:
		return NewI(0)
	case tensor.Int32:
		return NewI32(int32(0))
	case tensor.Int64:
		return NewI64(int64(0))
	case tensor.Byte:
		return NewU8(byte(0))
	case tensor.Bool:
		return NewB(false)
	default:
		panic("Unhandled dtype")
	}
}
