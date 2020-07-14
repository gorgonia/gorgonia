package xvm

import (
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type dummyTestOp struct {
	err error
}

/* Graph Building Related Methods */ // Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (u *dummyTestOp) Arity() int {
	panic("not implemented") // TODO: Implement
}

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (u *dummyTestOp) Type() hm.Type {
	panic("not implemented") // TODO: Implement
}

// returns the output shape as a function of the inputs
func (u *dummyTestOp) InferShape(_ ...gorgonia.DimSizer) (tensor.Shape, error) {
	panic("not implemented") // TODO: Implement
}

/* Machine related */ // executes the op
func (u *dummyTestOp) Do(v ...gorgonia.Value) (gorgonia.Value, error) {
	if u.err != nil {
		return nil, u.err
	}
	return v[0], nil
}

/* Analysis Related Methods */ // indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (u *dummyTestOp) ReturnsPtr() bool {
	panic("not implemented") // TODO: Implement
}

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (u *dummyTestOp) CallsExtern() bool {
	panic("not implemented") // TODO: Implement
}

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (u *dummyTestOp) OverwritesInput() int {
	panic("not implemented") // TODO: Implement
}

/* Other methods */
func (u *dummyTestOp) WriteHash(h hash.Hash) {
	panic("not implemented") // TODO: Implement
}

func (u *dummyTestOp) Hashcode() uint32 {
	panic("not implemented") // TODO: Implement
}

func (u *dummyTestOp) String() string {
	panic("not implemented") // TODO: Implement
}
