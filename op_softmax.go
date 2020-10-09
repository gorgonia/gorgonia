package gorgonia

import (
	"fmt"
	"hash"
	"math"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type softmaxOp struct {
}

func newSoftmaxOp(inputShape tensor.Shape) *softmaxOp {
	softmaxop := &softmaxOp{}

	return softmaxop
}

// Softmax -  implements the softmax operation described here: http://proceedings.mlr.press/v48/martins16.pdf
// Current implementation only supports float64
func Softmax(x *Node) (*Node, error) {
	xShape := x.Shape()
	op := newSoftmaxOp(xShape)

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

	var output interface{}

	switch arr := inputTensor.Data().(type) {
	case []float64:
		output = float64softMax(arr)
	case []float32:
		output = float32softMax(arr)
	default:
		return nil, fmt.Errorf("Softmax needs either []float32 or []float64, got %T", arr)
	}

	return tensor.New(tensor.Of(inputTensor.Dtype()), tensor.WithShape(inputTensor.Size()), tensor.WithEngine(inputTensor.Engine()), tensor.WithBacking(output)), nil
}

// FIXME: go2
func float64softMax(arr []float64) interface{} {
	output := make([]float64, len(arr))
	sum := 0.0

	for i, v := range arr {
		exp := math.Exp(v)
		sum += exp

		output[i] = exp
	}

	for i := range output {
		output[i] /= sum
	}

	return output
}

func float32softMax(arr []float32) interface{} {
	output := make([]float32, len(arr))
	sum := float32(0.0)

	for i, v := range arr {
		exp := float32(math.Exp(float64(v)))
		sum += exp

		output[i] = exp
	}

	for i := range output {
		output[i] /= sum
	}

	return output
}

type softmaxDiffOp struct {
}

func newSoftmaxOpDiff() *softmaxDiffOp {
	return &softmaxDiffOp{}
}

func (op *softmaxDiffOp) Arity() int {
	return 2
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

	return hm.NewFnType(ta, ta, ta) // f(float64, float64) float64
}

func (op *softmaxDiffOp) OverwritesInput() int { return -1 }

func (op *softmaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, err
	}

	var (
		in tensor.Tensor

		gradient tensor.Tensor
		ok       bool
	)

	switch t := inputs[0].(type) {
	case *dualValue:
		if in, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, errors.Errorf("input should be a tensor, got %T", inputs[0])
		}
	case tensor.Tensor:
		in = t
	default:
		return nil, nil, errors.Errorf("input type is not supported, got %T", inputs[0])
	}

	switch t := inputs[1].(type) {
	case *dualValue:
		if gradient, ok = t.Value.(tensor.Tensor); !ok {
			return nil, nil, errors.Errorf("gradient should be a tensor, got %T", inputs[1])
		}
	case tensor.Tensor:
		gradient = t
	default:
		return nil, nil, errors.Errorf("gradient type is not supported, got %T", inputs[1])
	}

	if in.Shape().Dims() != 1 || gradient.Shape().Dims() != 1 {
		return nil, nil, errors.Errorf("Expected input to have 1 dimensions")
	}

	return in, gradient, nil
}

func (op *softmaxDiffOp) Do(inputs ...Value) (Value, error) {
	inputTensor, gradTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SoftmaxDiff input: %w", err)
	}

	if inputTensor.Size() != gradTensor.Size() {
		return nil, fmt.Errorf("softmaxDiffOp.Do inputs sizes should be equal")
	}

	if !isFloat32Or64Array(inputTensor.Data()) {
		return nil, fmt.Errorf("softmaxDiffOp.Do expected input to be []float64 or []float32, got %T", inputTensor.Data())
	}

	if !isFloat32Or64Array(gradTensor.Data()) {
		return nil, fmt.Errorf("softmaxDiffOp.Do expected input to be []float64, got %T", gradTensor.Data())
	}

	input := inputTensor.Data().([]float64)
	value := gradTensor.Data().([]float64)

	output := make([]float64, len(input)*len(value))

	for i := 0; i < len(value); i++ {
		for j := 0; j < len(input); j++ {
			if i == j {
				output[i*j+j] = value[i] * (1 - input[i])
			} else {
				output[i*j+j] = -value[i] * input[i]
			}
		}
	}

	val := tensor.New(
		tensor.Of(inputTensor.Dtype()),
		tensor.WithShape(len(input), len(value)),
		tensor.WithEngine(inputTensor.Engine()),
		tensor.WithBacking(output), // FIXME
	)

	return val, nil
}

func isFloat32Or64Array(v interface{}) bool {
	if _, ok := v.([]float64); ok {
		return true
	}

	if _, ok := v.([]float32); ok {
		return true
	}

	return false
}

// ensure it complies with the Op interface
var (
	_ Op = &softmaxOp{}

	_ Op = &softmaxDiffOp{}
)
