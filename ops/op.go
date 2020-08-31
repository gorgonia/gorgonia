package ops

import (
	"fmt"
	"runtime/trace"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/shapes"
	"gorgonia.org/gorgonia/values"
)

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//		OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
//
// All Ops need to know somethings about themselves - there is no support for generic Ops.
type Op interface {
	/* Graph Building Related Methods */

	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
	Arity() int

	// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
	Type() hm.Type

	// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
	ShapeExpr() shapes.Expr

	/* Machine related */

	// Do executes the op.
	Do(...values.Value) (values.Value, error)

	/* Operational stuff */

	// Task generates a trace.Task for the Op. The Op is responsible for naming the task.
	//
	// In general it is best to just name it in the most general way possible.
	//
	// e.g. if a op is a reshape operation, then the standard implementation would have the
	// target shape as part of the name (e.g. "reshape(2,3)"). In this case it's best to just
	// call the task name "reshape", instead of "reshape(2,3)"
	Task() *trace.Task

	fmt.Stringer
}

// AnalyzableOp is any Op that provides enough intensionality for analysis during compilation phase.
type AnalyzableOp interface {
	Op

	// ReturnsPtr indicates if the Op will return a pointer (allowing possible inplace edits) or by value.
	// If it's false, the return value of the Op will be a copy of its input.
	ReturnsPtr() bool

	// CallsExtern informs if an op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool

	// OverwritesInput() is a method which states which input the output will be overwriting.
	// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
	// The method returns an int instead of a bool because potentially different operations may be allowed
	// to overwrite certain inputs. For example, consider an operation to increment a value:
	// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
	// the retVal of overwriteInput() will be 0 (inputs[0]).
	// -1 is returned if overwriting of input is disallowed
	OverwritesInput() int
}

// SDOp is any Op that supports symbolic differentiation
type SDOp interface {
	Op

	// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
	// returns []bool to indicate which input it is differentiable to
	DiffWRT(inputs int) []bool

	// SymDiff symbolically differentiates the op
	SymDiff(inputs gorgonia.Tensors, output, grad gorgonia.Tensor) (retVal gorgonia.Tensors, err error)
}

// ADOp is any Op that supports automatic differentiation. TODO
type ADOp interface {
	Op
	DoDiff() // TODO
}
