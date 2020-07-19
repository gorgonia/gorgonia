package xvm

import (
	"errors"
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type noOpTest struct {
	err error
}

/* Graph Building Related Methods */ // Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (u *noOpTest) Arity() int {
	panic("not implemented") // TODO: Implement
}

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (u *noOpTest) Type() hm.Type {
	panic("not implemented") // TODO: Implement
}

// returns the output shape as a function of the inputs
func (u *noOpTest) InferShape(_ ...gorgonia.DimSizer) (tensor.Shape, error) {
	panic("not implemented") // TODO: Implement
}

/* Machine related */ // executes the op
func (u *noOpTest) Do(v ...gorgonia.Value) (gorgonia.Value, error) {
	if u.err != nil {
		return nil, u.err
	}
	return v[0], nil
}

/* Analysis Related Methods */ // indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (u *noOpTest) ReturnsPtr() bool {
	panic("not implemented") // TODO: Implement
}

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (u *noOpTest) CallsExtern() bool {
	panic("not implemented") // TODO: Implement
}

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (u *noOpTest) OverwritesInput() int {
	panic("not implemented") // TODO: Implement
}

/* Other methods */
func (u *noOpTest) WriteHash(h hash.Hash) {
	panic("not implemented") // TODO: Implement
}

func (u *noOpTest) Hashcode() uint32 {
	panic("not implemented") // TODO: Implement
}

func (u *noOpTest) String() string {
	panic("not implemented") // TODO: Implement
}

type addOpF32Test struct{}

/* Graph Building Related Methods */ // Arity returns the number of inputs the Op expects. -1 indicates that it's n-ary and will be determined at runtime
func (a *addOpF32Test) Arity() int {
	return 2
}

// Informs the type of the Op (not the node). This will be used by the type system to infer the final type of the node
func (*addOpF32Test) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a, a)
}

// returns the output shape as a function of the inputs
func (a *addOpF32Test) InferShape(_ ...gorgonia.DimSizer) (tensor.Shape, error) {
	return tensor.ScalarShape(), nil
}

/* Machine related */ // executes the op
func (a *addOpF32Test) Do(vs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(vs) != 2 {
		return nil, errors.New("bad Arity")
	}
	res := gorgonia.F32(vs[0].Data().(float32) + vs[1].Data().(float32))
	return &res, nil
}

/* Analysis Related Methods */ // indicates if the Op will return a pointer (allowing possible inplace edits) or by value
// if it's false, the return value of the Op will be a copy of its input
func (a *addOpF32Test) ReturnsPtr() bool {
	return false
}

// Does this op potentially call external (cgo or cuda) functions (thereby requiring extra overhead for Go's trampolining thing)
func (a *addOpF32Test) CallsExtern() bool {
	return false
}

// overwriteInput() is a method which states which input the output will be overwriting.
// This allows for some efficiency gains as the underlying arrays wouldn't have to be re-allocated.
// The method returns an int instead of a bool because potentially different operations may be allowed
// to overwrite certain inputs. For example, consider an operation to increment a value:
// the IncrementOp would be a unary operator, and assuming we would like to overwrite the input,
// the retVal of overwriteInput() will be 0 (inputs[0]).
// -1 is returned if overwriting of input is disallowed
func (a *addOpF32Test) OverwritesInput() int {
	return -1
}

/* Other methods */
func (a *addOpF32Test) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "addOpF32Test")
}

func (a *addOpF32Test) Hashcode() uint32 {
	return 0
}

func (a *addOpF32Test) String() string {
	return "addOpF32"
}
