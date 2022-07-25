package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Applies SoftMax to the input x
func SoftMax(x *Node, axes ...int) (*Node, error) {
	op := newSoftmaxOp(x.Shape(), axes...)

	return ApplyOp(op, x)
}

// Applies LogSoftMax to the input x
func LogSoftMax(x *Node, axes ...int) (*Node, error) {
	op := newSoftmaxOp(x.Shape(), axes...)
	op.isLog = true

	return ApplyOp(op, x)
}

type softmaxOp struct {
	shape tensor.Shape
	axis  int
	isLog bool
}

func newSoftmaxOp(inputShape tensor.Shape, axes ...int) *softmaxOp {
	axis := -1
	if len(axes) > 0 {
		axis = axes[0]
	}
	softmaxop := &softmaxOp{
		shape: inputShape,
		axis:  axis,
	}

	return softmaxop
}

func (op *softmaxOp) Arity() int { return 1 }

func (op *softmaxOp) ReturnsPtr() bool { return false }

func (op *softmaxOp) CallsExtern() bool { return false }

func (op *softmaxOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "Softmax{%v}()", op.axis) }

func (op *softmaxOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxOp) String() string { return fmt.Sprintf("Softmax{%d, %v}()", op.axis, op.isLog) }

func (op *softmaxOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (op *softmaxOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a) // f(float64) float64
}

func (op *softmaxOp) OverwritesInput() int { return -1 }

func (op *softmaxOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var (
		in tensor.Tensor
		ok bool
	)

	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	return in, nil
}

func (op *softmaxOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Softmax input: %w", err)
	}

	aShape := inputTensor.Shape()
	ret := tensor.New(tensor.WithShape(aShape.Clone()...), tensor.Of(inputTensor.Dtype()))

	return op.UsePreallocDo(ret, inputTensor)

}

func (op *softmaxOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Softmax input: %w", err)
	}

	aShape := inputTensor.Shape()
	axis := aShape.Dims() - 1 // default: last dim

	if aShape.IsColVec() || (aShape.IsVector() && !aShape.IsRowVec()) {
		axis = 0
	}
	if op.axis != -1 {
		axis = op.axis
	}

	if !op.isLog {
		_, err = tensor.SoftMax(inputTensor, axis, tensor.WithReuse(prealloc.(tensor.Tensor)), tensor.UseUnsafe())
		if err != nil {
			return nil, err
		}
	} else {
		_, err = tensor.LogSoftMax(inputTensor, axis, tensor.WithReuse(prealloc.(tensor.Tensor)), tensor.UseUnsafe())
		if err != nil {
			return nil, err
		}
	}

	return prealloc, nil
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *softmaxOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("SoftmaxOp.DoDiff needs 1 arguments")
	}

	odv := output.boundTo.(*dualValue)
	idv := inputs[0].boundTo.(*dualValue)
	idvd := idv.d.(*tensor.Dense)
	diffOp := &softmaxDiffOp{op}

	result, err := diffOp.Do(idv.Value, odv.Value, odv.d)
	if err != nil {
		return err
	}

	sum, err := idvd.Add(result.(*tensor.Dense), tensor.UseUnsafe())
	if err != nil {
		return err
	}

	odv.d = sum

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *softmaxOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	diffOp := &softmaxDiffOp{op}
	nodes := make(Nodes, 1)

	nodes[0], err = ApplyOp(diffOp, inputs[0], output, grad)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *softmaxOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("softmax operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

type softmaxDiffOp struct {
	*softmaxOp
}

func (op *softmaxDiffOp) Arity() int { return 3 }

func (op *softmaxDiffOp) ReturnsPtr() bool { return false }

func (op *softmaxDiffOp) CallsExtern() bool { return false }

func (op *softmaxDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "SoftmaxDiff{%d, %v}()", op.axis, op.isLog)
}

func (op *softmaxDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxDiffOp) String() string {
	return fmt.Sprintf("SoftmaxDiff{%d, %v}()", op.axis, op.isLog)
}

func (op *softmaxDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *softmaxDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a, a) // f(float64) float64
}

func (op *softmaxDiffOp) OverwritesInput() int { return -1 }

func (op *softmaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, nil, err
	}

	var (
		in   tensor.Tensor
		out  tensor.Tensor
		grad tensor.Tensor
		ok   bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("input should be a tensor, got %T", inputs[0])
		}
	case tensor.Tensor:
		in = t
	default:
		return nil, nil, nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	switch t := inputs[1].(type) {
	case *dualValue:
		if out, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("output should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		out = t
	default:
		return nil, nil, nil, errors.Errorf("output type is not supported, got %T", inputs[1])
	}

	switch t := inputs[2].(type) {
	case *dualValue:
		if grad, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, nil, errors.Errorf("grad should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		grad = t
	default:
		return nil, nil, nil, errors.Errorf("grad type is not supported, got %T", inputs[1])
	}

	return in, out, grad, nil
}

func (op *softmaxDiffOp) Do(inputs ...Value) (Value, error) {
	x, y, grad, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	ret := tensor.New(tensor.WithShape(y.Shape().Clone()...), tensor.Of(y.Dtype()))

	return op.UsePreallocDo(ret, x, y, grad)

}

func (op *softmaxDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	_, y, grad, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	if op.isLog {
		return tensor.LogSoftMaxB(y, grad, op.axis, tensor.WithReuse(prealloc.(tensor.Tensor)), tensor.UseUnsafe())
	}

	return tensor.SoftMaxB(y, grad, op.axis, tensor.WithReuse(prealloc.(tensor.Tensor)), tensor.UseUnsafe())
}

// ensure it complies with the Op interface
var (
	_ Op   = &softmaxOp{}
	_ ADOp = &softmaxOp{}
	_ SDOp = &softmaxOp{}

	_ Op = &softmaxDiffOp{}
)
