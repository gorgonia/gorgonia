package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

type diagFlatOp struct{}

/* Graph Building Related Methods */

// Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (op diagFlatOp) Arity() int { return 1 }

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (op diagFlatOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := hm.TypeVariable('a')
	T := makeTensorType(2, b)
	return hm.NewFnType(a, T)
}

// returns the output shape as a function of the inputs
func (op diagFlatOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	in := inputs[0].(tensor.Shape)
	return tensor.Shape{in.TotalSize(), in.TotalSize()}, nil
}

/* Machine related */ // executes the op
func (op diagFlatOp) Do(vals ...Value) (Value, error) {
	if err := checkArity(op, len(vals)); err != nil {
		return nil, err
	}

	T := vals[0].(tensor.Tensor)
	return tensor.New(tensor.AsDenseDiag(T.Data())), nil
}

/* Analysis Related Methods */

// indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (op diagFlatOp) ReturnsPtr() bool { return false }

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (op diagFlatOp) CallsExtern() bool { return false }

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (op diagFlatOp) OverwritesInput() int { return -1 }

/* Other methods */
func (op diagFlatOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "DiagFlatOp") }

func (op diagFlatOp) Hashcode() uint32 { return simpleHash(op) }

func (op diagFlatOp) String() string { return "DiagFlat" }
