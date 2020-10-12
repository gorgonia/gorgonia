package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type softmaxOp struct {
	shape tensor.Shape
	axes  []int
}

func newSoftmaxOp(inputShape tensor.Shape, axes ...int) *softmaxOp {
	softmaxop := &softmaxOp{
		shape: inputShape,
		axes:  axes,
	}

	return softmaxop
}

// SoftMax -  implements the softmax operation
func SoftMax(x *Node, axis ...int) (*Node, error) {
	xShape := x.Shape()
	op := newSoftmaxOp(xShape, axis...)

	return ApplyOp(op, x)
}

func (op *softmaxOp) Arity() int {
	return 1
}

func (op *softmaxOp) ReturnsPtr() bool { return false }

func (op *softmaxOp) CallsExtern() bool { return false }

func (op *softmaxOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Softmax{}()")
}

func (op *softmaxOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxOp) String() string {
	return fmt.Sprintf("Softmax{}()")
}

func (op *softmaxOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}

func (op *softmaxOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(1, a)

	return hm.NewFnType(t, t)
}

func (op *softmaxOp) OverwritesInput() int { return -1 }

func (op *softmaxOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var in tensor.Tensor
	var ok bool

	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 1 {
		return nil, errors.Errorf("Expected input to have 1 dimensions")
	}

	return in, nil
}

func (op *softmaxOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Softmax input: %w", err)
	}

	aShape := inputTensor.Shape()
	axis := aShape.Dims() - 1 // default: last dim
	if aShape.IsColVec() || (aShape.IsVector() && !aShape.IsRowVec()) {
		axis = 0
	}

	if len(op.axes) > 0 {
		if op.axes[0] >= axis+1 || op.axes[0] < 0 {
			return nil, errors.Errorf("Cannot perform SoftMax on axis %d. Input has shape %v", op.axes[0], aShape)
		}

		axis = op.axes[0]
	}

	exp, err := tensor.Exp(inputTensor)
	if err != nil {
		return nil, fmt.Errorf("error calculating exp for SoftMax: %w", err)
	}

	sum, err := tensor.Sum(exp, axis)
	if err != nil {
		return nil, fmt.Errorf("error calculating sum for SoftMax: %w", err)
	}

	div, err := tensor.Div(exp, sum)
	if err != nil {
		return nil, fmt.Errorf("error calculating div for SoftMax: %w", err)
	}

	return div, nil
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *softmaxOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("SoftmaxOp.DoDiff needs 1 arguments")
	}

	odv := output.boundTo.(*dualValue)
	odvd := odv.Value.(tensor.Tensor)
	diffOp := newSoftmaxOpDiff()

	result, err := diffOp.Do()
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
func (op *softmaxOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	diffOp := newSoftmaxOpDiff()
	nodes := make(Nodes, 1)

	nodes[0], err = ApplyOp(diffOp, output)

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
}

func newSoftmaxOpDiff() *softmaxDiffOp {
	return &softmaxDiffOp{}
}

func (op *softmaxDiffOp) Arity() int {
	return 1
}

func (op *softmaxDiffOp) ReturnsPtr() bool { return false }

func (op *softmaxDiffOp) CallsExtern() bool { return false }

func (op *softmaxDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "SoftmaxDiff{}()")
}

func (op *softmaxDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *softmaxDiffOp) String() string {
	return fmt.Sprintf("SoftmaxDiff{}()")
}

func (op *softmaxDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *softmaxDiffOp) Type() hm.Type {
	aType := hm.TypeVariable('a')

	ta := newTensorType(1, aType)

	return hm.NewFnType(ta, ta) // f(float64) float64
}

func (op *softmaxDiffOp) OverwritesInput() int { return -1 }

func (op *softmaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var (
		in tensor.Tensor

		ok bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(tensor.Tensor); !ok {
			return nil, errors.Errorf("input should be a tensor, got %T", inputs[0])
		}
	case tensor.Tensor:
		in = t
	default:
		return nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	return in, nil
}

func (op *softmaxDiffOp) Do(inputs ...Value) (Value, error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	diag := tensor.New(tensor.AsDenseDiag(inputTensor))

	sm := inputTensor.Clone().(tensor.Tensor)

	err = sm.Reshape(inputTensor.Shape().TotalSize(), 1)
	if err != nil {
		return nil, fmt.Errorf("softmaxDiffOp.Do error reshaping the value: %w", err)
	}

	smT := sm.Clone().(tensor.Tensor)

	err = smT.T()
	if err != nil {
		return nil, fmt.Errorf("softmaxDiffOp.Do error transposing the value: %w", err)
	}

	smDot, err := tensor.MatMul(sm, smT)
	if err != nil {
		return nil, fmt.Errorf("softmaxDiffOp.Do error calculating dot product: %w", err)
	}

	result, err := tensor.Sub(diag, smDot)
	if err != nil {
		return nil, fmt.Errorf("softmaxDiffOp.Do error calculating sub: %w", err)
	}

	return result, nil
}

// ensure it complies with the Op interface
var (
	_ Op   = &softmaxOp{}
	_ ADOp = &softmaxOp{}
	_ SDOp = &softmaxOp{}

	_ Op = &softmaxDiffOp{}
)
