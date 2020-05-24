package values

import (
	"bytes"
	"fmt"
	"reflect"
	"unsafe"

	"gorgonia.org/tensor"
)

var scalarShape = tensor.ScalarShape()

// Scalar represents a scalar(non-array-based) value. Do note that it's the pointers of the scalar types (F64, F32, etc) that implement
// the Scalar interface. The main reason is primarily due to optimizations with regards to memory allocation and copying for device interoperability.
type Scalar interface {
	Value
	isScalar() bool
}

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

func NewF64(v float64) *F64 { r := F64(v); return &r }
func NewF32(v float32) *F32 { r := F32(v); return &r }
func NewI(v int) *I         { r := I(v); return &r }
func NewI64(v int64) *I64   { r := I64(v); return &r }
func NewI32(v int32) *I32   { r := I32(v); return &r }
func NewU8(v byte) *U8      { r := U8(v); return &r }
func NewB(v bool) *B        { r := B(v); return &r }

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

/* Any() */
// Any() is a method that returns the Go type.

// Any returns the Go type for the given value
func (v *F64) Any() float64 { return float64(*v) }

// Any returns the Go type for the given value
func (v *F32) Any() float32 { return float32(*v) }

// Any returns the Go type for the given value
func (v *I) Any() int { return int(*v) }

// Any returns the Go type for the given value
func (v *I64) Any() int64 { return int64(*v) }

// Any returns the Go type for the given value
func (v *I32) Any() int32 { return int32(*v) }

// Any returns the Go type for the given value
func (v *U8) Any() byte { return byte(*v) }

// Any returns the Go type for the given value
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

func (v *F64) isScalar() bool { return true }
func (v *F32) isScalar() bool { return true }
func (v *I) isScalar() bool   { return true }
func (v *I64) isScalar() bool { return true }
func (v *I32) isScalar() bool { return true }
func (v *U8) isScalar() bool  { return true }
func (v *B) isScalar() bool   { return true }

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

/* Oner */

func (v *F64) One() { *v = 1 }
func (v *F32) One() { *v = 1 }
func (v *I) One()   { *v = 1 }
func (v *I64) One() { *v = 1 }
func (v *I32) One() { *v = 1 }
func (v *U8) One()  { *v = 1 }
func (v *B) One()   { *v = true }

/* OneValuer */

func (v *F64) OneValue() Value { return NewF64(1) }
func (v *F32) OneValue() Value { return NewF32(1) }
func (v *I) OneValue() Value   { return NewI(1) }
func (v *I64) OneValue() Value { return NewI64(1) }
func (v *I32) OneValue() Value { return NewI32(1) }
func (v *U8) OneValue() Value  { return NewU8(1) }
func (v *B) OneValue() Value   { return NewB(true) }

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
