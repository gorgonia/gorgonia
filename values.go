package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	ti "github.com/chewxy/gorgonia/tensor/i"
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// Value represents a value that Gorgonia accepts
type Value interface {
	Type() hm.Type
	Shape() types.Shape
	Size() int
	Dtype() Dtype
	Eq(other Value) bool
	Data() interface{}

	clone() (Value, error)
	zero() Value

	fmt.Formatter
}

type Valuer interface {
	Value() Value
}

type Zeroer interface {
	Value
	Zero()
}

type Setter interface {
	SetAll(interface{}) error
}

// Tensor is a Value. It wraps over types.Tensor
type Tensor struct {
	types.Tensor
}

func NewTensorValue(dt Dtype, shp ...int) Tensor {
	var t types.Tensor

	switch dt {
	case Float64:
		t = tf64.NewTensor(tf64.WithShape(shp...))
	case Float32:
		t = tf32.NewTensor(tf32.WithShape(shp...))
	case Int:
		t = ti.NewTensor(ti.WithShape(shp...))
	default:
		panic("Unhandled yet")
	}
	return Tensor{Tensor: t}
}

func FromTensor(t types.Tensor) Tensor {
	return Tensor{Tensor: t}
}

// lazy values - this allows for pooling of tensorTypes
func (t Tensor) Type() hm.Type {
	dt, dim := tensorInfo(t.Tensor)
	shp := t.Tensor.Shape()
	tt := newTensorType(dim, dt)
	tt.shape = shp
	return tt
}

func (t Tensor) Dtype() Dtype {
	return dtypeToDtype(t.Tensor.Dtype())
}

func (t Tensor) Shape() types.Shape { return t.Tensor.Shape() }

func (t Tensor) Eq(other Value) bool {
	ot, ok := other.(Tensor)
	if !ok {
		return false
	}
	return ot.Tensor.Eq(t.Tensor)
}

func (t Tensor) clone() (Value, error) {
	retVal := Tensor{
		Tensor: tensor.Clone(t.Tensor),
	}
	return retVal, nil
}

func (t Tensor) zero() Value {
	t.Tensor.Zero()
	return t
}

// Scalar is more of a class of values than value, but eh, we'll leave it as is
type Scalar struct {
	v interface{}
	t Dtype
}

func NewScalarValue(val interface{}) Scalar {
	var retVal Scalar
	retVal.v = val
	switch val.(type) {
	case float64:
		retVal.t = Float64
	case float32:
		retVal.t = Float32
	case int:
		retVal.t = Int
	case int64:
		retVal.t = Int64
	case int32:
		retVal.t = Int32
	case byte:
		retVal.t = Byte
	case bool:
		retVal.t = Bool
	default:
		panic("Unhandled type")
	}
	return retVal
}

func (s Scalar) Type() hm.Type      { return s.t }
func (s Scalar) Dtype() Dtype       { return s.t }
func (s Scalar) Shape() types.Shape { return types.ScalarShape() }
func (s Scalar) Size() int          { return 1 }
func (s Scalar) Data() interface{}  { return s.v }

func (s Scalar) Eq(other Value) bool {
	os, ok := other.(Scalar)
	if !ok {
		return false
	}

	return os.v == s.v && os.t == s.t
}

func (s Scalar) sanity() (err error) {
	switch s.v.(type) {
	case float64:
		if s.t != Float64 {
			return errors.Errorf("Type mismatch. Want Float64. Got %v instead", s.t)
		}
	case float32:
		if s.t != Float32 {
			return errors.Errorf("Type mismatch. Want Float32. Got %v instead", s.t)
		}
	case int:
		if s.t != Int {
			return errors.Errorf("Type mismatch. Want Int. Got %v instead", s.t)
		}
	case int64:
		if s.t != Int64 {
			return errors.Errorf("Type mismatch. Want Int64. Got %v instead", s.t)
		}
	case int32:
		if s.t != Int32 {
			return errors.Errorf("Type mismatch. Want Int32. Got %v instead", s.t)
		}
	case byte:
		if s.t != Byte {
			return errors.Errorf("Type mismatch. Want Byte. Got %v instead", s.t)
		}
	case bool:
		if s.t != Bool {
			return errors.Errorf("Type mismatch. Want Bool. Got %v instead", s.t)
		}
	default:
		return errors.Errorf("Scalar %v is of Unsupported type %T", s.v, s.v)
	}
	return
}

func (s Scalar) clone() (Value, error) {
	s2 := Scalar{
		v: s.v,
		t: s.t,
	}
	err := s2.sanity()
	return s2, err
}

func (s Scalar) zero() Value {
	cloned, err := s.clone()
	if err != nil {
		panic(err) // yes, it's a panic.
	}
	s2 := cloned.(Scalar)

	switch s.t {
	case Float64:
		s2.v = 0.0
	case Float32:
		s2.v = float32(0.0)
	case Int:
		s2.v = 0
	case Int64:
		s2.v = int64(0)
	case Int32:
		s2.v = int32(0)
	case Byte:
		s2.v = byte(0)
	case Bool:
		s2.v = false
	}
	return s2

}

func (s Scalar) Format(state fmt.State, c rune) {
	if fmter, ok := s.v.(fmt.Formatter); ok {
		fmter.Format(state, c)
	}
	fmt.Fprintf(state, "%v", s.v)
	return
}
