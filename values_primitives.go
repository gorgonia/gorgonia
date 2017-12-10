package gorgonia

import (
	"bytes"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

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

func newF64(v float64) *F64 { r := F64(v); return &r }
func newF32(v float32) *F32 { r := F32(v); return &r }
func newI(v int) *I         { r := I(v); return &r }
func newI64(v int64) *I64   { r := I64(v); return &r }
func newI32(v int32) *I32   { r := I32(v); return &r }
func newU8(v byte) *U8      { r := U8(v); return &r }
func newB(v bool) *B        { r := B(v); return &r }

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
func (v *F64) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *F32) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *I) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *I64) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *I32) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *U8) Data() interface{} { return v.any() }

// Data returns the original representation of the Value
func (v *B) Data() interface{} { return v.any() }

func (v *F64) any() float64 { return float64(*v) }
func (v *F32) any() float32 { return float32(*v) }
func (v *I) any() int       { return int(*v) }
func (v *I64) any() int64   { return int64(*v) }
func (v *I32) any() int32   { return int32(*v) }
func (v *U8) any() byte     { return byte(*v) }
func (v *B) any() bool      { return bool(*v) }

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

func anyToScalar(any interface{}) (Scalar, tensor.Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, at.Dtype()
	case float64:
		return newF64(at), Float64
	case float32:
		return newF32(at), Float32
	case int:
		return newI(at), Int
	case int32:
		return newI32(at), Int32
	case int64:
		return newI64(at), Int64
	case byte:
		return newU8(at), Byte
	case bool:
		return newB(at), Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled", any, any))
	}
}

func anyToValue(any interface{}) (val Value, t hm.Type, dt tensor.Dtype, err error) {
	switch a := any.(type) {
	case Value:
		val = a
		t = TypeOf(a)
		dt = a.Dtype()
		return
	case float64, float32, int, int64, int32, byte, bool:
		val, dt = anyToScalar(any)
		t = dt
		return
	case F64:
		return newF64(float64(a)), tensor.Float64, tensor.Float64, nil
	case F32:
		return newF32(float32(a)), tensor.Float32, tensor.Float32, nil
	case I:
		return newI(int(a)), tensor.Int, tensor.Int, nil
	case I64:
		return newI64(int64(a)), tensor.Int64, tensor.Int64, nil
	case I32:
		return newI32(int32(a)), tensor.Int32, tensor.Int32, nil
	case U8:
		return newU8(byte(a)), tensor.Uint8, tensor.Uint8, nil
	case B:
		return newB(bool(a)), tensor.Bool, tensor.Bool, nil
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

func one(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return newF64(float64(1))
	case tensor.Float32:
		return newF32(float32(1))
	case tensor.Int:
		return newI(1)
	case tensor.Int32:
		return newI32(int32(1))
	case tensor.Int64:
		return newI64(int64(1))
	case tensor.Byte:
		return newU8(byte(1))
	case tensor.Bool:
		return newB(true)
	default:
		panic("Unhandled dtype")
	}
}

func zero(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return newF64(float64(0))
	case tensor.Float32:
		return newF32(float32(0))
	case tensor.Int:
		return newI(0)
	case tensor.Int32:
		return newI32(int32(0))
	case tensor.Int64:
		return newI64(int64(0))
	case tensor.Byte:
		return newU8(byte(0))
	case tensor.Bool:
		return newB(false)
	default:
		panic("Unhandled dtype")
	}
}
