package gorgonia

import (
	"bytes"
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type Scalar interface {
	Value
	Any() interface{}
}

type F64 float64
type F32 float32
type I int
type I32 int32
type I64 int64
type U8 byte
type B bool

func (v F64) Shape() types.Shape { return scalarShape }
func (v F32) Shape() types.Shape { return scalarShape }
func (v I) Shape() types.Shape   { return scalarShape }
func (v I64) Shape() types.Shape { return scalarShape }
func (v I32) Shape() types.Shape { return scalarShape }
func (v U8) Shape() types.Shape  { return scalarShape }
func (v B) Shape() types.Shape   { return scalarShape }

func (v F64) Size() int { return 0 }
func (v F32) Size() int { return 0 }
func (v I) Size() int   { return 0 }
func (v I64) Size() int { return 0 }
func (v I32) Size() int { return 0 }
func (v U8) Size() int  { return 0 }
func (v B) Size() int   { return 0 }

func (v F64) Data() interface{} { return v.Any() }
func (v F32) Data() interface{} { return v.Any() }
func (v I) Data() interface{}   { return v.Any() }
func (v I64) Data() interface{} { return v.Any() }
func (v I32) Data() interface{} { return v.Any() }
func (v U8) Data() interface{}  { return v.Any() }
func (v B) Data() interface{}   { return v.Any() }

func (v F64) Any() interface{} { return float64(v) }
func (v F32) Any() interface{} { return float32(v) }
func (v I) Any() interface{}   { return int(v) }
func (v I64) Any() interface{} { return int64(v) }
func (v I32) Any() interface{} { return int32(v) }
func (v U8) Any() interface{}  { return byte(v) }
func (v B) Any() interface{}   { return bool(v) }

func (v F64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v F32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v I) Format(s fmt.State, c rune)   { formatScalar(v, s, c) }
func (v I64) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v I32) Format(s fmt.State, c rune) { formatScalar(v, s, c) }
func (v U8) Format(s fmt.State, c rune)  { formatScalar(v, s, c) }
func (v B) Format(s fmt.State, c rune)   { formatScalar(v, s, c) }

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
		case F64, F32, U8, B:
			buf.WriteRune('v')
		default:
			buf.WriteRune(c)
		}
	case 'f', 'g':
		switch v.(type) {
		case I, I64, I32, U8, B:
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

	fmt.Fprintf(s, buf.String(), v.Any())
}

func anyToScalar(any interface{}) (Scalar, Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, DtypeOf(at)
	case float64:
		return F64(at), Float64
	case float32:
		return F32(at), Float32
	case int:
		return I(at), Int
	case int32:
		return I32(at), Int32
	case int64:
		return I64(at), Int64
	case byte:
		return U8(at), Byte
	case bool:
		return B(at), Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled", any, any))
	}
}

func anyToValue(any interface{}) (val Value, t hm.Type, dt Dtype, err error) {
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
	case types.Tensor:
		val = a
		t = TypeOf(a)
		dt = DtypeOf(a)
		return
	default:
		err = errors.Errorf("value %v of %T not yet handled", any, any)
		return
	}
}

func one(dt Dtype) Scalar {
	switch dt {
	case Float64:
		return F64(1)
	case Float32:
		return F32(1)
	case Int:
		return I(1)
	case Int32:
		return I32(1)
	case Int64:
		return I64(1)
	case Byte:
		return U8(1)
	case Bool:
		return B(true)
	default:
		panic("Unhandled dtype")
	}
}

func zero(dt Dtype) Scalar {
	switch dt {
	case Float64:
		return F64(0)
	case Float32:
		return F32(0)
	case Int:
		return I(0)
	case Int32:
		return I32(0)
	case Int64:
		return I64(0)
	case Byte:
		return U8(0)
	case Bool:
		return B(false)
	default:
		panic("Unhandled dtype")
	}
}
