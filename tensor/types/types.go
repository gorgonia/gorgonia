package types

import (
	"fmt"
	"math"
)

type Dtype byte

const (
	Float64 Dtype = iota
	Float32
	Int
	Int64
	Int32
	Byte
	Bool

	MAXDTYPE
)

type Tensor interface {
	// info about the ndarray
	Info() *AP
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int

	// ops
	Reshape(...int) error
	T(axes ...int) error
	UT()

	// data related interface
	Zero()
	SetAll(interface{}) error
	Data() interface{}

	// Equality
	Eq(other Tensor) bool

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

// NormOrder represents the order of the norm. Ideally, we'd only represent norms with a uint/byte.
// But there are norm types that are outside numerical types, such as nuclear norm and fobenius norm.
// So it is internally represented by a float. If Go could use NaN and Inf as consts, it would have been best,
// Instead, we use constructors. Both Nuclear and Frobenius norm types are represented as NaNs
//
// The using of NaN and Inf as "special" Norm types lead to the need for IsInf() and IsFrobenius() and IsNuclear() method
type NormOrder float64

func Norm(ord int) NormOrder   { return NormOrder(float64(ord)) }
func InfNorm() NormOrder       { return NormOrder(math.Inf(1)) }
func NegInfNorm() NormOrder    { return NormOrder(math.Inf(-1)) }
func UnorderedNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000001)) }
func FrobeniusNorm() NormOrder { return NormOrder(math.Float64frombits(0x7ff8000000000002)) }
func NuclearNorm() NormOrder   { return NormOrder(math.Float64frombits(0x7ff8000000000003)) }

// Valid() is a helper method that deterines if the norm order is valid. A valid norm order is
// one where the fraction component is 0
func (n NormOrder) Valid() bool {
	switch {
	case math.IsNaN(float64(n)):
		nb := math.Float64bits(float64(n))
		if math.Float64bits(float64(UnorderedNorm())) == nb || math.Float64bits(float64(FrobeniusNorm())) == nb || math.Float64bits(float64(NuclearNorm())) == nb {
			return true
		}
	case math.IsInf(float64(n), 0):
		return true
	default:
		if _, frac := math.Modf(float64(n)); frac == 0.0 {
			return true
		}
	}
	return false
}

func (n NormOrder) IsUnordered() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(UnorderedNorm()))
}

func (n NormOrder) IsFrobenius() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(FrobeniusNorm()))
}

func (n NormOrder) IsNuclear() bool {
	return math.Float64bits(float64(n)) == math.Float64bits(float64(NuclearNorm()))
}

func (n NormOrder) IsInf(sign int) bool {
	return math.IsInf(float64(n), sign)
}

func (n NormOrder) String() string {
	switch {
	case n.IsUnordered():
		return "Unordered"
	case n.IsFrobenius():
		return "Frobenius"
	case n.IsNuclear():
		return "Nuclear"
	case n.IsInf(1):
		return "+Inf"
	case n.IsInf(-1):
		return "-Inf"
	default:
		return fmt.Sprintf("Norm %v", float64(n))
	}
	panic("unreachable")
}

type ConsOpt interface {
	Opt()
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
