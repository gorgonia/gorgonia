package values

import (
	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

type V interface {
	tensor.DescWithStorage
	tensor.DataSizer
	tensor.Memory
	tensor.Engineer
}

// Value represents a value that Gorgonia accepts. At this point it is implemented by:
//   - *dense.Dense[DT]
//   - *exprgraph.Value[DT, T]
//   - *dual.Value[DT,T]
//   - scalar.Scalar[DT
//
// A Value is essentially any thing that knows its own type and shape.
// Most importantly though, a Value is a pointer - and can be converted into a tensor.Memory.
// This is done for the sake of interoperability with external devices like cgo or CUDA or OpenCL.
// This also means for the most part most Values will be allocated on the heap.
// There are some performance tradeoffs made in this decision, but ultimately this is better than having to manually manage blocks of memory
type Value[DT any] interface {
	tensor.Basic[DT]
}

// Typer is anything that knows its own type
type Typer interface {
	Type() hm.Type
}

// Valuer is any type that can return a Value.
type Valuer[DT any] interface {
	Value() Value[DT]
}

// Zeroer is a Value that can zero itself (here zero is used as a verb).
type Zeroer[DT any] interface {
	Value[DT]
	// Zero()
}

// ZeroValuer is a a Value that can provide the zero-value of its type.
type ZeroValuer[DT any, T Value[DT]] interface {
	Value[DT]
	ZeroValue() T
}

// Oner is a Value that can set itself to 1.
type Oner[DT any] interface {
	Value[DT]
	One()
}

// OneValuer is a value that can provide the one-value of its type.
type OneValuer[DT any] interface {
	Value[DT]
	OneValue() Value[DT]
}

// ScalarOner is anything that can get the equivalent value to 1.
type ScalarOner[DT any] interface{ ScalarOne() DT }

// ScalarZero is any datatype that can get the equivalent of the value zero.
// This interface is not strictly necessary if one defines one's datatypes well -
// that is to say, make the zeroth value useful. Bue occasionally, there may be
// data types where the zeroth value is not zero. So it may be useful to implement this.
type ScalarZeroer[DT any] interface{ ScalarZero() DT }

// ValueEqualer represents any type that can perform a equal value check
type ValueEqualer[DT any] interface {
	ValueEq(Value[DT]) bool
}

// ValueCloser represents any type that can perform a close-value check
type ValueCloser[DT float32 | float64, T Value[DT]] interface {
	ValueClose(T) bool
}

// Cloner represents any type that can clone itself.
type Cloner[T any] interface {
	Clone() T
}

// CloneErrorer represents any type that can clone itself and return an error if necessary
type CloneErrorer[T any] interface {
	Clone() (T, error)
}

// CopierTo represents any type that can copy data to the destination.
type CopierTo[T any] interface {
	CopyTo(dest T) error
}

// CopierFrom represents any type that can copy data from the source provided.
type CopierFrom interface {
	CopyFrom(src any) error
}

type ShallowCloner[T any] interface {
	ShallowClone() T
}
