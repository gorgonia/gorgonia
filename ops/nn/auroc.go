package nnops

import (
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type auroc struct {
	margin float64
	power  float64
}

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (op *auroc) Arity() int { return 2 }

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op *auroc) Type() hm.Type {
	panic("not implemented") // TODO: Implement
}

// returns the output shape as a function of the inputs
func (op *auroc) InferShape(_ ...gorgonia.DimSizer) (tensor.Shape, error) {
	panic("not implemented") // TODO: Implement
}

// Do executes the op
func (op *auroc) Do(vals ...gorgonia.Value) (gorgonia.Value, error) {
	pred := vals[0]
	correct := vals[1]
	// TODO: check that they are vectors

	isTrue := make([]bool, correct.Shape()[0]) // TODO: check if rowVec or colVec. Don't just assume
	cData := correct.Data()
	switch d := cData.(type) {
	case []float64:
		for i := range d {
			isTrue[i] = d[i] > 0 // any value > 0 is considered the true class
		}
	case []float32:
		for i := range d {
			isTrue[i] = d[i] > 0
		}
	}

	// some fuckery around the graph - we want to skip these nodes actually, otherwise the function wouldn't be symbolically differentiated.
	for i, iIsTrue := range isTrue {
		for j, jIsTrue := range isTrue {
			if i == j {
				continue
			}
			if jIsTrue {
				continue
			}
			if !iIsTrue {
				continue
			}

			// Slice? What other methods are unsafe and differentiable? TODO

			// obj := fn(pred1, pred0, op.power, op.margin)
			// also note: fn should be an inverse of R1 in Jason's original implementation, if I read the paper correctly.
		}
	}

}

// ReturnsPtr indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (op *auroc) ReturnsPtr() bool {
	panic("not implemented") // TODO: Implement
}

// CallsExtern checks if this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (op *auroc) CallsExtern() bool {
	panic("not implemented") // TODO: Implement
}

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (op *auroc) OverwritesInput() int {
	panic("not implemented") // TODO: Implement
}

/* Other methods */
func (op *auroc) WriteHash(h hash.Hash) {
	panic("not implemented") // TODO: Implement
}

func (op *auroc) Hashcode() uint32 {
	panic("not implemented") // TODO: Implement
}

func (op *auroc) String() string {
	panic("not implemented") // TODO: Implement
}

// DiffWRT indicates if the op is differentiable with regards to the given number of inputs
// returns []bool to indicate which input it is differentiable to
func (op *auroc) DiffWRT(inputs int) []bool { return []bool{true, true} }

// SymDiff symbolically differentiates the op
func (op *auroc) SymDiff(inputs gorgonia.Nodes, output *gorgonia.Node, grad *gorgonia.Node) (retVal gorgonia.Nodes, err error) {
	panic("not implemented") // TODO: Implement
}

func (op *auroc) DoDiff(ctx gorgonia.ExecutionContext, inputs gorgonia.Nodes, output *gorgonia.Node) error {
	panic("not implemented") // TODO: Implement
}

type aurocDiff struct{ *auroc }

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (op *aurocDiff) Arity() int {
	panic("not implemented") // TODO: Implement
}

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op *aurocDiff) Type() hm.Type {
	panic("not implemented") // TODO: Implement
}

// returns the output shape as a function of the inputs
func (op *aurocDiff) InferShape(_ ...gorgonia.DimSizer) (tensor.Shape, error) {
	panic("not implemented") // TODO: Implement
}

/* Machine related */ // executes the op
func (op *aurocDiff) Do(_ ...gorgonia.Value) (gorgonia.Value, error) {
	panic("not implemented") // TODO: Implement
}

/* Analysis Related Methods */ // indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (op *aurocDiff) ReturnsPtr() bool {
	panic("not implemented") // TODO: Implement
}

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (op *aurocDiff) CallsExtern() bool {
	panic("not implemented") // TODO: Implement
}

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (op *aurocDiff) OverwritesInput() int {
	panic("not implemented") // TODO: Implement
}

/* Other methods */
func (op *aurocDiff) WriteHash(h hash.Hash) {
	panic("not implemented") // TODO: Implement
}

func (op *aurocDiff) Hashcode() uint32 {
	panic("not implemented") // TODO: Implement
}

func (op *aurocDiff) String() string {
	panic("not implemented") // TODO: Implement
}
