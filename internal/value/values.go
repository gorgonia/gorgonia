package value

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/internal/constructor"
	"gorgonia.org/tensor"
)

// START_DEF_VALUE OMIT

// Value represents a value that Gorgonia accepts. At this point it is implemented by:
//		- all scalar value types (F64, F32... etc)
// 		- *tensor.Dense
// 		- *dualValue
//
// A Value is essentially any thing that knows its own type and shape.
// Most importantly though, a Value is a pointer - and can be converted into a tensor.Memory.
// This is done for the sake of interoperability with external devices like cgo or CUDA or OpenCL.
// This also means for the most part most Values will be allocated on the heap.
// There are some performance tradeoffs made in this decision, but ultimately this is better than having to manually manage blocks of memory
type Value interface {
	Shape() tensor.Shape // Shape  returns the shape of the Value. Scalar values return ScalarShape()
	Size() int           // Size represents the number of elements in the Value. Note that in cases such as a *tensor.Dense, the underlying slice MAY have more elements than the Size() reports. This is correct.
	Data() interface{}   // Data returns the original representation of the Value
	Dtype() tensor.Dtype // Dtype returns the Dtype of the value

	tensor.Memory
	fmt.Formatter
}

// END_DEF_VALUE OMIT

// Valuer is any type that can return a Value
type Valuer interface {
	Value() Value
}

// Zeroer is a Value that can zero itself
type Zeroer interface {
	Value
	Zero()
}

// ZeroValuer is a a Value that can provide the zero-value of its type
type ZeroValuer interface {
	Value
	ZeroValue() Value
}

// Dtyper represents any type (typically a Value) that knows its own Dtype
type Dtyper interface {
	Dtype() tensor.Dtype
}

// Typer represents any type (typically a Op) that knows its own Type
type Typer interface {
	Type() hm.Type
}

// ValueEqualer represents any type that can perform a equal value check
type ValueEqualer interface {
	ValueEq(Value) bool
}

// ValueCloser represents any type that can perform a close-value check
type ValueCloser interface {
	ValueClose(interface{}) bool
}

// Cloner represents any type that can clone itself.
type Cloner interface {
	Clone() interface{}
}

// CloneErrorer represents any type that can clone itself and return an error if necessary
type CloneErrorer interface {
	Clone() (interface{}, error)
}

// CopierTo represents any type that can copy data to the destination.
type CopierTo interface {
	CopyTo(dest interface{}) error
}

// CopierFrom represents any type that can copy data from the source provided.
type CopierFrom interface {
	CopyFrom(src interface{}) error
}

// ValueGrad is any type that has a value and a grad. This is used for Solvers
type ValueGrad interface {
	Valuer
	Grad() (Value, error)
}

// TypeOf returns the Type of the value
func TypeOf(v Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return constructor.MakeTensorType(dim, dt)
	case Scalar:
		return t.Dtype()
	case Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

// Scalar represents a scalar(non-array-based) value. Do note that it's the pointers of the scalar types (F64, F32, etc) that implement
// the Scalar interface. The main reason is primarily due to optimizations with regards to memory allocation and copying for device interoperability.
type Scalar interface {
	Value
	IsScalar() bool
}
