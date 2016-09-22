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

// FunctionFlag are flags for calling Tensor functions. Use only with FuncOpt
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

// FuncOpt are optionals for calling Tensor function.
type FuncOpt func() (FunctionFlag, interface{})

// WithIncr passes in a Tensor to be incremented.
func WithIncr(incr Tensor) FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return Incr, incr
	}
	return f
}

// WithReuse passes in a Tensor to be reused.
func WithReuse(reuse Tensor) FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return Reuse, reuse
	}
	return f
}

// UseSafe ensures that the operation is a safe operation (copies data, does not clobber). This is the default option for most methods and functions
func UseSafe() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return SafeOp, nil
	}
	return f
}

// UseUnsafe ensures that the operation is an unsafe operation - data will be clobbered, and operations performed inplace
func UseUnsafe() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return UnsafeOp, nil
	}
	return f
}

// AsSameType makes sure that the return Tensor is the same type as input Tensors.
func AsSameType() FuncOpt {
	f := func() (FunctionFlag, interface{}) {
		return AsSame, nil
	}
	return f
}
