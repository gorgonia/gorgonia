package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type byIndicesOp struct {
	axis int
}

func newByIndicesOp(axis int) *byIndicesOp {
	if axis < 0 {
		axis = 0
	}

	return &byIndicesOp{
		axis: axis,
	}
}

// ByIndices is an operation that takes the indices as input and return the selected values from those indices.
// The default axis in 0
func ByIndices(x *Node, indices *Node, axis int) (*Node, error) {
	op := newByIndicesOp(axis)

	return ApplyOp(op, x, indices)
}

func (op *byIndicesOp) Arity() int { return 2 }

func (op *byIndicesOp) ReturnsPtr() bool { return false }

func (op *byIndicesOp) CallsExtern() bool { return false }

func (op *byIndicesOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

func (op *byIndicesOp) Hashcode() uint32 { return simpleHash(op) }

func (op *byIndicesOp) String() string {
	return fmt.Sprintf("ByIndicesOp{axis=%d}", op.axis)
}

func (op *byIndicesOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	i := inputs[1].(tensor.Shape).Clone()
	if !i.IsVectorLike() {
		return nil, errors.Errorf("Expected indices to be a vector-like. Got %v instead", i)
	}

	s[op.axis] = i.TotalSize()

	return s, nil
}

func (op *byIndicesOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := makeTensorType(1, tensor.Int)

	return hm.NewFnType(a, b, a)
}

func (op *byIndicesOp) OverwritesInput() int { return -1 }

func (op *byIndicesOp) checkInput(inputs ...Value) (x, indices tensor.Tensor, err error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, err
	}

	var ok bool
	if x, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("Expected input to be a tensor, got %T", inputs[0])
	}
	if indices, ok = inputs[1].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("Expected indices to be a tensor. Got %T instead", inputs[1])
	}

	if indices.Dtype() != tensor.Int {
		return nil, nil, errors.Errorf("Expected indices to have tensor.Int as a Dtype. Got %T instead", indices.Dtype())
	}

	return x, indices, nil
}

func (op *byIndicesOp) Do(inputs ...Value) (Value, error) {
	inputTensor, indices, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check ByIndicesOp input: %w", err)
	}

	return tensor.ByIndices(inputTensor, indices, op.axis)
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *byIndicesOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 2 {
		return fmt.Errorf("byIndicesOp.DoDiff needs 2 arguments")
	}

	odv := output.boundTo.(*dualValue)
	odvd := odv.Value.(tensor.Tensor)

	diffOp := &ByIndicesOpDiffOp{op}

	result, err := diffOp.Do(inputs[0].boundTo, inputs[1].boundTo)
	if err != nil {
		return err
	}

	err = result.(*tensor.Dense).Reshape(odvd.Shape()...)
	if err != nil {
		return err
	}

	sum, err := odvd.(*tensor.Dense).Add(result.(*tensor.Dense), tensor.UseUnsafe())
	if err != nil {
		return err
	}

	odv.d = sum

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *byIndicesOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	x := inputs[0]
	indices := inputs[1]

	diffOp := &ByIndicesOpDiffOp{op}
	nodes := make(Nodes, op.Arity())

	nodes[0], err = ApplyOp(diffOp, x, grad, indices)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *byIndicesOp) DiffWRT(inputs int) []bool {
	if inputs != op.Arity() {
		panic(fmt.Sprintf("ByIndicesOp operator needs %d inputs, got %d instead", op.Arity(), inputs))
	}

	return []bool{true, false}
}

type ByIndicesOpDiffOp struct {
	*byIndicesOp
}

func (op *ByIndicesOpDiffOp) Arity() int { return 3 }

func (op *ByIndicesOpDiffOp) ReturnsPtr() bool { return false }

func (op *ByIndicesOpDiffOp) CallsExtern() bool { return false }

func (op *ByIndicesOpDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op *ByIndicesOpDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *ByIndicesOpDiffOp) String() string {
	return fmt.Sprintf("ByIndicesOpDiff{}(%d)", op.axis)
}

func (op *ByIndicesOpDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *ByIndicesOpDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := makeTensorType(1, tensor.Int)

	return hm.NewFnType(a, a, b, a)
}

func (op *ByIndicesOpDiffOp) OverwritesInput() int { return -1 }

func (op *ByIndicesOpDiffOp) checkInput(inputs ...Value) (in, indices, gradient *tensor.Dense, err error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, nil, err
	}

	var (
		ok bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(*tensor.Dense); !ok {
			return nil, nil, nil, errors.Errorf("input should be a tensor.Tensor, got %T", inputs[0])
		}
	case *tensor.Dense:
		in = t
	default:
		return nil, nil, nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	switch t := inputs[2].(type) {
	case *dualValue:
		if gradient, ok = t.Value.(*tensor.Dense); !ok {
			return nil, nil, nil, errors.Errorf("gradient should be a tensor, got %T", inputs[2])
		}
	case *tensor.Dense:
		gradient = t
	default:
		return nil, nil, nil, errors.Errorf("gradient type is not supported, got %T", inputs[2])
	}

	switch t := inputs[1].(type) {
	case *tensor.Dense:
		indices = t
	default:
		return nil, nil, nil, errors.Errorf("indices type %T is not supported", inputs[1])
	}

	return in, indices, gradient, nil
}

func (op *ByIndicesOpDiffOp) Do(inputs ...Value) (Value, error) {
	inputTensor, gradTensor, indices, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check ByIndicesOpDiff input: %w", err)
	}

	output, err := tensor.ByIndicesB(inputTensor, gradTensor, indices, op.axis)
	if err != nil {
		return nil, err
	}

	return output, nil
}

// ensure it complies with the Op interface
var (
	_ Op = &ByIndicesOpDiffOp{}

	_ Op   = &byIndicesOp{}
	_ SDOp = &byIndicesOp{}
	_ ADOp = &byIndicesOp{}
)
