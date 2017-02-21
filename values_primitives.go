package gorgonia

import (
	"bytes"
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type Scalar interface {
	Value
	isScalar() bool
}

type F64 float64
type F32 float32
type I int
type I32 int32
type I64 int64
type U8 byte
type B bool

func newF64(v float64) *F64 { r := F64(v); return &r }
func newF32(v float32) *F32 { r := F32(v); return &r }
func newI(v int) *I         { r := I(v); return &r }
func newI64(v int64) *I64   { r := I64(v); return &r }
func newI32(v int32) *I32   { r := I32(v); return &r }
func newU8(v byte) *U8      { r := U8(v); return &r }
func newB(v bool) *B        { r := B(v); return &r }

func (v *F64) Shape() tensor.Shape { return scalarShape }
func (v *F32) Shape() tensor.Shape { return scalarShape }
func (v *I) Shape() tensor.Shape   { return scalarShape }
func (v *I64) Shape() tensor.Shape { return scalarShape }
func (v *I32) Shape() tensor.Shape { return scalarShape }
func (v *U8) Shape() tensor.Shape  { return scalarShape }
func (v *B) Shape() tensor.Shape   { return scalarShape }

func (v *F64) Size() int { return 0 }
func (v *F32) Size() int { return 0 }
func (v *I) Size() int   { return 0 }
func (v *I64) Size() int { return 0 }
func (v *I32) Size() int { return 0 }
func (v *U8) Size() int  { return 0 }
func (v *B) Size() int   { return 0 }

func (v *F64) Data() interface{} { return v.Any() }
func (v *F32) Data() interface{} { return v.Any() }
func (v *I) Data() interface{}   { return v.Any() }
func (v *I64) Data() interface{} { return v.Any() }
func (v *I32) Data() interface{} { return v.Any() }
func (v *U8) Data() interface{}  { return v.Any() }
func (v *B) Data() interface{}   { return v.Any() }

func (v *F64) Any() float64 { return float64(*v) }
func (v *F32) Any() float32 { return float32(*v) }
func (v *I) Any() int       { return int(*v) }
func (v *I64) Any() int64   { return int64(*v) }
func (v *I32) Any() int32   { return int32(*v) }
func (v *U8) Any() byte     { return byte(*v) }
func (v *B) Any() bool      { return bool(*v) }

func (v *F64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v *F32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v *I) Format(s fmt.State, c rune)   { formatScalar(v, s, c) }
func (v *I64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v *I32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v *U8) Format(s fmt.State, c rune)  { formatScalar(v, s, c) }
func (v *B) Format(s fmt.State, c rune)   { formatScalar(v, s, c) }

func (v *F64) Dtype() tensor.Dtype { return tensor.Float64 }
func (v *F32) Dtype() tensor.Dtype { return tensor.Float32 }
func (v *I) Dtype() tensor.Dtype   { return tensor.Int }
func (v *I64) Dtype() tensor.Dtype { return tensor.Int64 }
func (v *I32) Dtype() tensor.Dtype { return tensor.Int32 }
func (v *U8) Dtype() tensor.Dtype  { return tensor.Byte }
func (v *B) Dtype() tensor.Dtype   { return tensor.Bool }

func (v *F64) isScalar() bool { return true }
func (v *F32) isScalar() bool { return true }
func (v *I) isScalar() bool   { return true }
func (v *I64) isScalar() bool { return true }
func (v *I32) isScalar() bool { return true }
func (v *U8) isScalar() bool  { return true }
func (v *B) isScalar() bool   { return true }

func (v *F64) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }
func (v *F32) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }
func (v *I) Uintptr() uintptr   { return uintptr(unsafe.Pointer(v)) }
func (v *I64) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }
func (v *I32) Uintptr() uintptr { return uintptr(unsafe.Pointer(v)) }
func (v *U8) Uintptr() uintptr  { return uintptr(unsafe.Pointer(v)) }
func (v *B) Uintptr() uintptr   { return uintptr(unsafe.Pointer(v)) }

// MemSize

func (v *F64) MemSize() uintptr { return 8 }
func (v *F32) MemSize() uintptr { return 4 }
func (v *I) MemSize() uintptr   { return reflect.TypeOf(*v).Size() }
func (v *I64) MemSize() uintptr { return 8 }
func (v *I32) MemSize() uintptr { return 4 }
func (v *U8) MemSize() uintptr  { return 1 }
func (v *B) MemSize() uintptr   { return reflect.TypeOf(*v).Size() }

func (v *F64) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }
func (v *F32) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }
func (v *I) Pointer() unsafe.Pointer   { return unsafe.Pointer(v) }
func (v *I64) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }
func (v *I32) Pointer() unsafe.Pointer { return unsafe.Pointer(v) }
func (v *U8) Pointer() unsafe.Pointer  { return unsafe.Pointer(v) }
func (v *B) Pointer() unsafe.Pointer   { return unsafe.Pointer(v) }

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
		s.Write([]byte(DtypeOf(v).String()))
		s.Write([]byte{' '})
	}

	fmt.Fprintf(s, buf.String(), v.Data())
}

func anyToScalar(any interface{}) (Scalar, tensor.Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, DtypeOf(at)
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
		dt = DtypeOf(a)
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
		dt = DtypeOf(a)
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
