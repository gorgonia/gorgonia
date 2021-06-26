package gorgonia

import (
	"fmt"
	"hash"
	"log"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type randomGenF func() float64

type dropoutOp struct {
	probability float64
	isTraining  bool
	rndGen      randomGenF
}

func newDropoutOp(probability float64, rndGen randomGenF) *dropoutOp {
	dropoutop := &dropoutOp{
		probability: probability,
		isTraining:  true,
		rndGen:      rndGen,
	}

	return dropoutop
}

func (op *dropoutOp) SetTraining(isTraining bool) error { op.isTraining = isTraining; return nil }

func (op *dropoutOp) Arity() int { return 1 }

func (op *dropoutOp) ReturnsPtr() bool { return false }

func (op *dropoutOp) CallsExtern() bool { return false }

func (op *dropoutOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, op.String()) }

func (op *dropoutOp) Hashcode() uint32 { return simpleHash(op) }

func (op *dropoutOp) String() string { return fmt.Sprintf("Dropout{training=%v}()", op.isTraining) }

func (op *dropoutOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (op *dropoutOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a) // f(float64) float64
}

func (op *dropoutOp) OverwritesInput() int { return -1 }

func (op *dropoutOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
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

func (op *dropoutOp) Do(inputs ...Value) (retVal Value, err error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Dropout input: %w", err)
	}

	if op.probability == 0.0 || !op.isTraining {
		return inputTensor, nil
	}

	ret := tensor.New(tensor.WithShape(inputTensor.Shape().Clone()...), tensor.Of(inputTensor.Dtype()))

	op.do(inputTensor.Data(), ret.Data())

	return ret, nil
}

func (op *dropoutOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	inputTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Dropout input: %w", err)
	}

	if op.probability == 0.0 || !op.isTraining {
		return inputTensor, nil
	}

	op.do(inputTensor.Data, prealloc.Data())

	return prealloc, nil
}

func (op *dropoutOp) do(input, output interface{}) {
	keepProb := 1.0 - op.probability

	switch v := input.(type) {
	case []float32:
		outputV := output.([]float32)
		for i, d := range v {
			r := float32(op.rndGen())
			if r < float32(keepProb) {
				outputV[i] = d / float32(keepProb)
			}
		}
	case []float64:
		outputV := output.([]float64)
		for i, d := range v {
			r := op.rndGen()
			if r < keepProb {
				outputV[i] = d / keepProb
			}
		}
	default:
		log.Panicf("unknown dtype: %T", output)
	}
}

// DoDiff calculates the diff and sets its value to the output node. Implementation for ADOp interface.
func (op *dropoutOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("dropout diff requires 1 arguments")
	}

	odv := output.boundTo.(*dualValue)
	// idv := inputs[0].boundTo.(*dualValue)

	switch data := odv.d.Data().(type) {
	case []float32:
		for i := range data {
			data[i] /= float32(odv.d.Size())
		}
	case []float64:
		for i := range data {
			data[i] /= float64(odv.d.Size())
		}
	}

	return nil
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *dropoutOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	var totalSize interface{}

	switch grad.Dtype() {
	case tensor.Float32:
		totalSize = float32(inputs[0].Shape().TotalSize())
	case tensor.Float64:
		totalSize = float64(inputs[0].Shape().TotalSize())
	}

	c := NewConstant(totalSize)
	div, err := Div(grad, c)
	if err != nil {
		return nil, fmt.Errorf("division failed: %w", err)
	}

	return Nodes{div}, nil
}

// DiffWRT is an implementation for the SDOp interface
func (op *dropoutOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("dropout operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

// ensure it complies with the Op interface
var (
	_ Op          = &dropoutOp{}
	_ ADOp        = &dropoutOp{}
	_ SDOp        = &dropoutOp{}
	_ TrainModeOp = &dropoutOp{}
)
