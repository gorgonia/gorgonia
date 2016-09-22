package types

import "fmt"

type Dtype byte

const (
	Float64 Dtype = iota
	Float32
	Int
	Int64
	Int32
	Bool

	MAXDTYPE
)

type Tensor interface {
	// info about the ndarray
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int

	// ops
	Reshape(...int) error
	T(axes ...int) error
	Zero()

	// Equality
	Eq(other Tensor) bool

	Data() interface{}

	// type overloading shit
	IsScalar() bool
	ScalarValue() interface{}

	// view related shit
	IsView() bool
	Materialize() Tensor

	fmt.Formatter
	fmt.Stringer
}

type Slice interface {
	Start() int
	End() int
	Step() int
}

type FunctionFlag byte

const (
	SafeOp FunctionFlag = iota
	UnsafeOp
	Reuse
	Incr
	AsSame
	AsTensorF64
	AsTensorF32
	AsTensorInt
)

type FuncOpt func() (FunctionFlag, interface{})

func WithIncr(incr Tensor) FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return Incr, incr
	}
	return f
}

func WithReuse(reuse Tensor) FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return Reuse, reuse
	}
	return f
}

func UseSafe() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return SafeOp, nil
	}
	return f
}

func UseUnsafe() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return UnsafeOp, nil
	}
	return f
}

func AsSameType() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return AsSame, nil
	}
	return f
}
