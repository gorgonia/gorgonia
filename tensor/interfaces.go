package tensor

import (
	"reflect"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
)

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

// A Number is any type that can perform arithmetics. This allows for extensibility
type Number interface {
	Add(Number) (Number, error)
	Sub(Number) (Number, error)
	Mul(Number) (Number, error)
	Div(Number) (Number, error)
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

	headerer
	arrayer
	setAP(*AP)
	rtype() reflect.Type
	ostrides() []int
	oshape() Shape
	transposeAxes() []int
	transposeIndex(i int, transposePat, strides []int) int
	len() int
}

type SparseTensor interface {
	Sparse
	AsCSC()
	AsCSR()
	Indices() []int
	Indptr() []int

	headerer
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
}
