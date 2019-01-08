package ops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// DimSizer is any type (typically a tensor.Shape) that allows querying for a dimension size given an input dimension.
type DimSizer interface {
	DimSize(int) (int, error)
}

// ShapesToDimSizers is a convenience function to convert a slice of tensor.Shape to a slice of DimSizer
func ShapesToDimSizers(shapes []tensor.Shape) []DimSizer {
	retVal := make([]DimSizer, len(shapes))
	for i, s := range shapes {
		retVal[i] = s
	}
	return retVal
}

// DimSizersToShapes is a convenience function to convert a slice of DimSizer to a slice of tensor.Shape. It will return an error if any of them isn't a tensor.Shape
func DimSizersToShapes(ds []DimSizer) ([]tensor.Shape, error) {
	retVal := make([]tensor.Shape, len(ds))
	var ok bool
	for i, d := range ds {
		if retVal[i], ok = d.(tensor.Shape); !ok {
			return nil, errors.Errorf("Dimsizer %d is not a Shape.", i)
		}
	}
	return retVal, nil
}

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//		OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
type Op interface {
	/* Graph Building Related Methods */

	Arityer

	// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
	Type() hm.Type

	// returns the output shape as a function of the inputs
	InferShape(...DimSizer) (tensor.Shape, error)

	/* Machine related */

	// executes the op
	Do(...value.Value) (value.Value, error)

	/* Analysis Related Methods */

	// indicates if the Op will return a pointer (allowing possible inplace edits) or by value
	// if it's false, the return value of the Op will be a copy of its input
	ReturnsPtr() bool

	// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool

	// overwriteInput() is a method which states which input the output will be overwriting.
	// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
	// The method returns an int instead of a bool because potentially different operations may be allowed
	// to overwrite certain inputs. For example, consider an operation to increment a value:
	// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
	// the retVal of overwriteInput() will be 0 (inputs[0]).
	// -1 is returned if overwriting of input is disallowed
	OverwritesInput() int

	/* Other methods */
	WriteHash(h hash.Hash)
	Hashcode() uint32
	fmt.Stringer
}

// A UnaryOp is an Op that takes only one input
type UnaryOp interface {
	Op

	IsUnary() bool
}

// A BinaryOp is an Op that takes only two inputs
type BinaryOp interface {
	Op

	IsBinary() bool
}

// BestDoer ...
type BestDoer interface {
	Op

	BestDo(prealloc value.Value, vals ...value.Value) (value.Value, error)
}

// A NoRetOp is an Op that reads a value, but does not return any value. It's a representation of a not-pure function
type NoRetOp interface {
	Op

	ReturnsNothing() bool
}

// ReductionOp changes the shape of the node
type ReductionOp interface {
	Op

	IsReduction() bool
}

// IncrDoer increments the toIncr with the result of doing
type IncrDoer interface {
	IncrDo(toIncr value.Value, inputs ...value.Value) error
}

// UsePreallocDoer is an op that works when a preallocated value is provided
type UsePreallocDoer interface {
	UsePreallocDo(prealloc value.Value, inputs ...value.Value) (value.Value, error)
}

// UnsafeDoer is an op that will overwrite the underlying value.
type UnsafeDoer interface {
	UnsafeDo(inputs ...value.Value) (value.Value, error)
}

// CUDADoer uses CUDA to perform the Op.
type CUDADoer interface {
	CUDADo(extern execution.External, dev execution.Device, prealloc value.Value, inputs ...value.Value) (retVal value.Value, err error)
}

// CLDoer uses OpenCL to perform the Op. As of now, there are NO Ops that support OpenCL
type CLDoer interface {
	CLDo(inputs ...value.Value) (value.Value, error)
}

//Arityer is any object that knows its arity
type Arityer interface {
	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
	Arity() int
}
