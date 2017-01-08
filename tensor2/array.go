package tensor

import "github.com/pkg/errors"

// An Array is a representation of a backing
type Array interface {
	Len() int          // returns the length of the array
	Cap() int          // returns the cap of the array
	Data() interface{} // returns as its original self

	Get(i int) interface{}          // Gets the value at index i
	Set(i int, v interface{}) error // Sets the value at index i to the value

	Slice(Slice) (Array, error)

	Eq
	Zeroer
	MemSetter
}

func makeArray(dt Dtype, size int) Array {
	switch dt {
	case Float64:
		return make(f64s, size, size)
	case Float32:
		return make(f32s, size, size)
	case Int:
		return make(ints, size, size)
	case Int64:
		return make(i64s, size, size)
	case Int32:
		return make(i32s, size, size)
	case Byte:
		return make(u8s, size, size)
	case Bool:
		return make(bs, size, size)
	}

	if am, ok := dt.(ArrayMaker); ok {
		return am.MakeArray(size)
	}
	panic("Unsupported Dtype")
}

func arrayFromInterface(a interface{}) Array {
	switch at := a.(type) {
	case Array:
		return at
	case []float64:
		return f64s(at)
	case []float32:
		return f32s(at)
	case []int:
		return ints(at)
	case []int64:
		return i64s(at)
	case []int32:
		return i32s(at)
	case []byte:
		return u8s(at)
	case []bool:
		return bs(at)
	}
	panic("Unreachable")
}

type f64s []float64
type f32s []float32
type ints []int
type i64s []int64
type i32s []int32
type u8s []byte
type bs []bool

/* BASIC ARRAY TYPE HANDLING */

// Float64ser is any array that can turn into a []float64
type Float64ser interface {
	Float64s() []float64
}

// Float64ser is any array that can turn into a []float32
type Float32ser interface {
	Float32s() []float32
}

// Float64ser is any array that can turn into a []int
type Intser interface {
	Ints() []int
}

// Int64ser is any array that can turn into a []int64
type Int64ser interface {
	Int64s() []int64
}

// Int32ser is any array that can turn into a []int32
type Int32ser interface {
	Int32s() []int32
}

// Byteser is any array that can turn into a []byte
type Byteser interface {
	Bytes() []byte
}

// Boolser is any array that can turn into a []bool
type Boolser interface {
	Bools() []bool
}

// Dtyper is for any array implementation that knows its own Dtype
type Dtyper interface {
	Dtype() Dtype
}

/* OTHER TYPE CLASSES */

// Number is any array where you can perform basic arithmetic on. The arithmethic methods are expected to clober the value of the receiver
type Number interface {
	Array
	Add(Number) error
	Sub(Number) error
	Mul(Number) error
	Div(Number) error
}

type SafeNumber interface {
	Number
	SafeAdd(Number) (Array, error)
	SafeSub(Number) (Array, error)
	SafeMul(Number) (Array, error)
	SafeDiv(Number) (Array, error)
}

// Float is any type where you can perform floating point operations. Arrays that also implement Float will have linalg performed
type Float interface {
	Number
	HasNaN() bool
	HasInf() bool
}

// ElemEq is any array type that you can perform elementwise equality on
type ElEq interface {
	Array
	Oner

	ElEq(other ElEq, same bool) (Array, error)
}

// ElOrd is any type where you can perform an ordered comparison
type ElOrd interface {
	ElEq
	Lt(other ElOrd, same bool) (Array, error)
	Lte(other ElOrd, same bool) (Array, error)
	Gt(other ElOrd, same bool) (Array, error)
	Gte(other ElOrd, same bool) (Array, error)
}

func copyArray(dest, src Array) (int, error) {
	if cf, ok := dest.(CopierFrom); ok {
		return cf.CopyFrom(src)
	}

	if ct, ok := src.(CopierTo); ok {
		return ct.CopyTo(dest)
	}

	return 0, errors.Errorf("Unable to copy %v to %v", src, dest)
}

func typeOf(a Array) (Dtype, error) {
	switch at := a.(type) {
	case f64s:
		return Float64, nil
	case f32s:
		return Float32, nil
	case ints:
		return Int, nil
	case i64s:
		return Int64, nil
	case i32s:
		return Int32, nil
	case u8s:
		return Byte, nil
	case bs:
		return Bool, nil

	case Dtyper:
		return at.Dtype(), nil
	}

	return nil, errors.Errorf("Array %T has no known Dtype", a)
}
