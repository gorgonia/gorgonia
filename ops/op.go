package ops

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
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
	Do(ctx context.Context, vs ...values.Value) (retVal values.Value, err error)

	/* Operational stuff */

	fmt.Stringer
}

// PreAllocOp represents and Op that has a PreallocDo() method. The PreallocDo method is exactly the same as Do() except it also requres a previously preallocated value.
type PreallocOp interface {
	Op

	// PreallocDo performs the Op with the return value passed in as a preallocated value.
	PreallocDo(ctx context.Context, prealloc values.Value, vs ...values.Value) (retVal values.Value, err error)
}

// AnalyzableOp is any Op that provides enough intensionality for analysis during compilation phase.
type AnalyzableOp interface {
	Op

	// CallsExtern informs if an op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool
}

/*
// SDOp is any Op that supports symbolic differentiation
type SDOp interface {
	Op

	// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
	// returns []bool to indicate which input it is differentiable to
	DiffWRT(inputs int) []bool

	// SymDiff symbolically differentiates the op
	SymDiff(inputs gorgonia.Tensors, output, grad gorgonia.Tensor) (retVal gorgonia.Tensors, err error)
}
*/

// ADOp is any Op that supports automatic differentiation. TODO
type ADOp interface {
	Op
	DoDiff() // TODO
}
