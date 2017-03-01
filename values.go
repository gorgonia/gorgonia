package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
)

// Value represents a value that Gorgonia accepts. At this point it is implemented by:
//		- all scalar value types (F64, F32... etc)
// 		- *tensor.Dense
// 		- *dualValue
type Value interface {
	Shape() tensor.Shape // Shape  returns the shape of the Value. Scalar values return ScalarShape()
	Size() int           // Size represents the number of elements in the Value. Note that in cases such as a *tensor.Dense, the underlying slice MAY have more elements than the Size() reports. This is correct.
	Data() interface{}   // Data returns the original representation of the Value
	Dtype() tensor.Dtype // Dtype returns the Dtype of the value

	Memory
	fmt.Formatter
}

// Memory is a representation of memory of the value
type Memory interface {
	Uintptr() uintptr
	MemSize() uintptr
	Pointer() unsafe.Pointer
}

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

// Cloner represents any type that can clone itself.
type Cloner interface {
	Clone() interface{}
}

// CopierTo represents any type that can copy data to the destination.
type CopierTo interface {
	CopyTo(dest interface{}) error
}

// CopierFrom represents any type that can copy data from the source provided.
type CopierFrom interface {
	CopyFrom(src interface{}) error
}

// Setter is a any value that can Memset itself to the provided value
// type Setter interface {
// 	SetAll(interface{}) error
// }
