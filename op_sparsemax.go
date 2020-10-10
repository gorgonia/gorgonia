package gorgonia

import (
	"fmt"
	"hash"
	"sort"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type sparsemaxOp struct {
}

func newSparsemaxOp() *sparsemaxOp {
	sparsemaxop := &sparsemaxOp{}

	return sparsemaxop
}

// Sparsemax -  implements the sparsemax operation described here: http://proceedings.mlr.press/v48/martins16.pdf
func Sparsemax(x *Node) (*Node, error) {
	op := newSparsemaxOp()

	return ApplyOp(op, x)
}

func (op *sparsemaxOp) Arity() int {
	return 1
}

func (op *sparsemaxOp) ReturnsPtr() bool { return false }

func (op *sparsemaxOp) CallsExtern() bool { return false }

func (op *sparsemaxOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Sparsemax{}()")
}

func (op *sparsemaxOp) Hashcode() uint32 { return simpleHash(op) }

func (op *sparsemaxOp) String() string {
	return fmt.Sprintf("Sparsemax{}()")
}

func (op *sparsemaxOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}

func (op *sparsemaxOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(1, a)

	return hm.NewFnType(t, t)
}

func (op *sparsemaxOp) OverwritesInput() int { return -1 }

func (op *sparsemaxOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var in tensor.Tensor
	var ok bool

	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor, got %T", inputs[0])
	}

	if in.Shape().Dims() != 1 {
		return nil, errors.Errorf("Expected input to have 1 dimensions")
	}

	return in, nil
}

func (op *sparsemaxOp) Do(inputs ...Value) (Value, error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Sparsemax input: %w", err)
	}

	var output interface{}

	switch inputTensor.Dtype() {
	case tensor.Float64:
		output = op.float64sparseMax(inputTensor)
	case tensor.Float32:
		output = op.float32sparseMax(inputTensor)
	default:
		return nil, fmt.Errorf("invalid input type for Sparsemax, expected float64 or float32, got: %v", inputTensor.Dtype())
	}

	return tensor.New(tensor.Of(inputTensor.Dtype()), tensor.WithShape(inputTensor.Size()), tensor.WithEngine(inputTensor.Engine()), tensor.WithBacking(output)), nil
}

// FIXME: go2 generics
func (op *sparsemaxOp) float32sparseMax(inputTensor tensor.Tensor) interface{} {
	sortedData := make([]float32, inputTensor.Size())

	copy(sortedData, inputTensor.Data().([]float32))

	sort.Slice(sortedData, func(i, j int) bool {
		return sortedData[i] > sortedData[j]
	})

	kArray := make([]float32, len(sortedData))
	cumArray := make([]float32, len(sortedData))
	cumSum := float32(0.0)
	maxIndex := 0

	for i := 0; i < len(sortedData); i++ {
		kArray[i] = 1 + float32(i)*sortedData[i]
		cumSum += sortedData[i]

		cumArray[i] = cumSum - sortedData[i]

		if kArray[i] > cumArray[i] {
			maxIndex = i + 1
		}
	}

	threshold := float32(cumArray[maxIndex-1]-1) / float32(maxIndex)
	output := make([]float32, inputTensor.Size())

	for i := 0; i < inputTensor.Size(); i++ {
		v, _ := inputTensor.At(i)
		vF := v.(float32)

		if vF-threshold > 0 {
			output[i] = vF - threshold
		}
	}

	return output
}

func (op *sparsemaxOp) float64sparseMax(inputTensor tensor.Tensor) interface{} {
	sortedData := make([]float64, inputTensor.Size())

	copy(sortedData, inputTensor.Data().([]float64))

	sort.Slice(sortedData, func(i, j int) bool {
		return sortedData[i] > sortedData[j]
	})

	kArray := make([]float64, len(sortedData))
	cumArray := make([]float64, len(sortedData))
	cumSum := 0.0
	maxIndex := 0

	for i := 0; i < len(sortedData); i++ {
		kArray[i] = 1 + float64(i)*sortedData[i]
		cumSum += sortedData[i]

		cumArray[i] = cumSum - sortedData[i]

		if kArray[i] > cumArray[i] {
			maxIndex = i + 1
		}
	}

	threshold := float64(cumArray[maxIndex-1]-1) / float64(maxIndex)
	output := make([]float64, inputTensor.Size())

	for i := 0; i < inputTensor.Size(); i++ {
		v, _ := inputTensor.At(i)
		vF := v.(float64)

		if vF-threshold > 0 {
			output[i] = vF - threshold
		}
	}

	return output
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *sparsemaxOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 2 {
		return fmt.Errorf("SparsemaxOp.DoDiff needs 2 arguments")
	}

	odv := output.boundTo.(*dualValue)
	odvd := odv.Value.(tensor.Tensor)
	diffOp := &sparsemaxDiffOp{}

	result, err := diffOp.Do(inputs[0].boundTo, inputs[1].boundTo)
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
func (op *sparsemaxOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	t := inputs[0]

	diffOp := &sparsemaxDiffOp{}
	nodes := make(Nodes, 1)

	nodes[0], err = ApplyOp(diffOp, t, grad)

	return nodes, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *sparsemaxOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("sparsemax operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

type sparsemaxDiffOp struct {
}

func newSparsemaxOpDiff() *sparsemaxDiffOp {
	return &sparsemaxDiffOp{}
}

func (op *sparsemaxDiffOp) Arity() int {
	return 2
}

func (op *sparsemaxDiffOp) ReturnsPtr() bool { return false }

func (op *sparsemaxDiffOp) CallsExtern() bool { return false }

func (op *sparsemaxDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "SparsemaxDiff{}()")
}

func (op *sparsemaxDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *sparsemaxDiffOp) String() string {
	return fmt.Sprintf("SparsemaxDiff{}()")
}

func (op *sparsemaxDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *sparsemaxDiffOp) Type() hm.Type {
	aType := hm.TypeVariable('a')

	ta := newTensorType(1, aType)

	return hm.NewFnType(ta, ta, ta) // f(float64, float64) float64
}

func (op *sparsemaxDiffOp) OverwritesInput() int { return -1 }

func (op *sparsemaxDiffOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, error) {
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
			return nil, nil, errors.Errorf("input should be a tensor.Tensor, got %T", inputs[0])
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

func (op *sparsemaxDiffOp) Do(inputs ...Value) (Value, error) {
	inputTensor, gradTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check SparsemaxDiff input: %w", err)
	}

	if inputTensor.Size() != gradTensor.Size() {
		return nil, fmt.Errorf("sparsemaxDiffOp.Do inputs sizes should be equal")
	}

	var diff interface{}

	switch inputTensor.Dtype() {
	case tensor.Float64:
		outputData, ok := gradTensor.Data().([]float64)
		if !ok {
			return nil, fmt.Errorf("sparsemaxDiffOp.Do expected input to be []float64, got %T", inputTensor.Data())
		}

		diff = op.float64sparseMaxDiff(inputTensor.Data().([]float64), outputData)
	case tensor.Float32:
		outputData, ok := gradTensor.Data().([]float32)
		if !ok {
			return nil, fmt.Errorf("sparsemaxDiffOp.Do expected input to be []float32, got %T", inputTensor.Data())
		}

		diff = op.float32sparseMaxDiff(inputTensor.Data().([]float32), outputData)
	default:
		return nil, fmt.Errorf("sparsemaxDiffOp.Do expected input to be []float64 or []float32, got %T", inputTensor.Data())
	}

	val := tensor.New(
		tensor.Of(inputTensor.Dtype()),
		tensor.WithShape(inputTensor.Size()),
		tensor.WithEngine(inputTensor.Engine()),
		tensor.WithBacking(diff),
	)

	return val, nil
}

// FIXME: go2 generics
func (op *sparsemaxDiffOp) float32sparseMaxDiff(data, outputData []float32) interface{} {
	nonZeros := float32(0.0)
	inputSum := float32(0.0)
	diff := make([]float32, len(data))

	for i, v := range data {
		if v == 0.0 {
			continue
		}

		diff[i] = 1.0

		inputSum += outputData[i]
		nonZeros++
	}

	sum := float32(0.0)

	if nonZeros > 0 {
		sum = inputSum / nonZeros
	}

	for i := range diff {
		diff[i] *= (outputData[i] - sum)
	}

	return diff
}

func (op *sparsemaxDiffOp) float64sparseMaxDiff(data, outputData []float64) interface{} {
	nonZeros := 0.0
	inputSum := 0.0
	diff := make([]float64, len(data))

	for i, v := range data {
		if v == 0.0 {
			continue
		}

		diff[i] = 1.0

		inputSum += outputData[i]
		nonZeros++
	}

	sum := 0.0

	if nonZeros > 0 {
		sum = inputSum / nonZeros
	}

	for i := range diff {
		diff[i] *= (outputData[i] - sum)
	}

	return diff
}

// ensure it complies with the Op interface
var (
	_ Op = &sparsemaxDiffOp{}

	_ Op   = &sparsemaxOp{}
	_ SDOp = &sparsemaxOp{}
	_ ADOp = &sparsemaxOp{}
)
