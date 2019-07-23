package op

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
	"gorgonia.org/gorgonia/values"
)

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//		OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
type Op interface {
	/* Graph Building Related Methods */

	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
	Arity() int

	// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
	Type() hm.Type

	// returns the output shape as a function of the inputs
	InferShape(...DimSizer) (tensor.Shape, error)

	/* Machine related */

	// executes the op
	Do(...values.Value) (values.Value, error)

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
