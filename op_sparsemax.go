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

func newSparsemaxOp(inputShape tensor.Shape) *sparsemaxOp {
	sparsemaxop := &sparsemaxOp{}

	return sparsemaxop
}

// Sparsemax -  implements the sparsemax operation described here: http://proceedings.mlr.press/v48/martins16.pdf
// Current implementation only supports float64
func Sparsemax(x *Node) (*Node, error) {
	xShape := x.Shape()
	op := newSparsemaxOp(xShape)

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
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 1 {
		return nil, errors.Errorf("Expected input to have 1 dimensions")
	}

	return in, nil
}

func (op *sparsemaxOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Sparsemax input: %w", err)
	}

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

	return tensor.New(tensor.Of(inputTensor.Dtype()), tensor.WithShape(inputTensor.Size()), tensor.WithEngine(inputTensor.Engine()), tensor.WithBacking(output)), nil
}

// ensure it complies with the Op interface
var _ Op = &sparsemaxOp{}
