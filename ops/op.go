package ops

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
)

// Desc represents a description of an operation
type Desc interface {
	/* Graph Building Related Methods */

	// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime.
	Arity() int

	// Type informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node.
	Type() hm.Type

	// ShapeExpr informs the shape operations that the Op will do. A quick primer is given in the README of the shapes package.
	ShapeExpr() shapes.Expr

	fmt.Stringer
}

func GetOp[DT any, T values.Value[DT]](op Desc) (Op[DT, T], error) { panic("NYI") }

// An Op is a symbolic representation of an operation
// Think of them as functions, taking an input (or multiple), and outputting something
//
// All Ops have type signatures that look like this:
//
//	OpName :: (Floats a) ⇒ Tensor a → Tensor a → Tensor a
//
// All Ops need to know somethings about themselves - there is no support for generic Ops.
type Op[DT any, T values.Value[DT]] interface {
	Desc

	// Do executes the op.
	Do(ctx context.Context, vs ...T) (retVal T, err error)
}

// HKOp is a special kind of op
type HKOp[DT1, DT2 any, T values.Value[DT1], U values.Value[DT2]] interface {
	Desc
	Do(ctx context.Context, vs ...T) (retVal U, err error)
}

// PreallocOp represents and Op that has a PreallocDo() method. The PreallocDo method is exactly the same as Do() except it also requres a previously preallocated value.
type PreallocOp[DT any, T values.Value[DT]] interface {
	Op[DT, T]

	// PreallocDo performs the Op with the return value passed in as a preallocated value.
	PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal T, err error)
}

// PreallocHKOp represents and Op that has a PreallocDo() method. The PreallocDo method is exactly the same as Do() except it also requres a previously preallocated value.
type PreallocHKOp[DT1, DT2 any, T values.Value[DT1], U values.Value[DT2]] interface {
	HKOp[DT1, DT2, T, U]

	PreallocDo(ctx context.Context, prealloc T, vs ...T) (retVal U, err error)
}

// AnalyzableOp is any Op that provides enough intensionality for analysis during compilation phase.
type AnalyzableOp[DT any, T values.Value[DT]] interface {
	Op[DT, T]

	// CallsExtern informs if an op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
	CallsExtern() bool
}

// Operand represents an operand that an Op operates on. In short, it's a gorgonia.Tensor.
type Operand interface {
	Shape() shapes.Shape
}

// Statement represents an Op that is not necessarily pure. It's more akin to a programming statement, and may have side effects.
type Statement interface {
	IsStatement()
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


// ADOp is any Op that supports automatic differentiation
type ADOp interface {
	Op
	DoDiff(ctx context.Context, inputs []datatypes.Tensor, output datatypes.Tensor) error
}
*/
