package values

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

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

// Typer is anything that knows its own type
type Typer interface {
	Type() hm.Type
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
