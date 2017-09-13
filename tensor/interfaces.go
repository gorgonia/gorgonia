package tensor

import (
	"reflect"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
)

// Dtyper is any type that has a Dtype
type Dtyper interface {
	Dtype() Dtype
}

// Eq is any type where you can perform an equality test
type Eq interface {
	Eq(interface{}) bool
}

// Cloner is any type that can clone itself
type Cloner interface {
	Clone() interface{}
}

// Dataer is any type that returns the data in its original form (typically a Go slice of something)
type Dataer interface {
	Data() interface{}
}

// Boolable is any type has a zero and one value, and is able to set itself to either
type Boolable interface {
	Zeroer
	Oner
}

// A Zeroer is any type that can set itself to the zeroth value. It's used to implement the arrays
type Zeroer interface {
	Zero()
}

// A Oner is any type that can set itself to the equivalent of one. It's used to implement the arrays
type Oner interface {
	One()
}

// A MemSetter is any type that can set itself to a value.
type MemSetter interface {
	Memset(interface{}) error
}

// A Densor is any type that can return a *Dense
type Densor interface {
	Dense() *Dense
}

// ScalarRep is any Tensor that can represent a scalar
type ScalarRep interface {
	IsScalar() bool
	ScalarValue() interface{}
}

// View is any Tensor that can provide a view on memory
type View interface {
	Tensor
	IsView() bool
	IsMaterializable() bool
	Materialize() Tensor
}

// Slicer is any tensor that can slice
type Slicer interface {
	Slice(...Slice) (View, error)
}

// DenseTensor is the interface for any Dense tensor.
type DenseTensor interface {
	Tensor
	Info() *AP

	DataOrder() DataOrder
	IsVector() bool
	IsMatrix() bool

	// headerer
	// arrayer
	unsafeMem
	setAP(*AP)
	rtype() reflect.Type
	reshape(dims ...int) error

	isTransposed() bool
	ostrides() []int
	oshape() Shape
	transposeAxes() []int
	transposeIndex(i int, transposePat, strides []int) int
	oldAP() *AP
	setOldAP(ap *AP)
	parentTensor() *Dense
	setParentTensor(*Dense)
	len() int
	cap() int

	// operations
	Inner(other Tensor) (retVal interface{}, err error)
	MatMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error)
	MatVecMul(other Tensor, opts ...FuncOpt) (retVal *Dense, err error)
	TensorMul(other Tensor, axesA, axesB []int) (retVal *Dense, err error)
	stackDense(axis int, others ...DenseTensor) (DenseTensor, error)
}

type SparseTensor interface {
	Sparse
	AsCSC()
	AsCSR()
	Indices() []int
	Indptr() []int

	// headerer
}

type MaskedTensor interface {
	DenseTensor
	IsMasked() bool
	SetMask([]bool)
	Mask() []bool
}

// Kinder. Bueno.
type Kinder interface {
	Kind() reflect.Kind
}

type headerer interface {
	hdr() *storage.Header
}

type arrayer interface {
	arr() array
	arrPtr() *array
}

type unsafeMem interface {
	Set(i int, x interface{})
	GetF64(i int) float64
	GetF32(i int) float32
	Float64s() []float64
	Float32s() []float32
}
