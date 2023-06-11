package gorgonia

import (
	"fmt"
	"hash"
	"log"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type embeddingOp struct {
}

func (op embeddingOp) Arity() int {
	return 2
}

func (op embeddingOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a) // f(float64) float64
}

func (op embeddingOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Embedding requires 2 inputs, got %d", len(inputs))
	}
	embeddingShape := inputs[0].(tensor.Shape)
	if embeddingShape.Dims() != 2 {
		return nil, fmt.Errorf("Embedding requires a 2D embedding matrix, got %dD", embeddingShape.Dims())
	}
	inputShape := inputs[1].(tensor.Shape)
	outputShape := []int{}
	outputShape = append(outputShape, inputShape...)
	outputShape = append(outputShape, embeddingShape[1])
	return outputShape, nil
}

func (op embeddingOp) Do(value ...Value) (Value, error) {
	embddingTensor, inputTensor, err := op.checkInput(value...)
	if err != nil {
		return nil, fmt.Errorf("Can't check Embedding input: %w", err)
	}
	outputShape, err := op.InferShape(embddingTensor.Shape(), inputTensor.Shape())
	if err != nil {
		return nil, fmt.Errorf("Can't infer Embedding output shape: %w", err)
	}
	ret := tensor.New(tensor.WithShape(outputShape...), tensor.Of(embddingTensor.Dtype()))
	op.do(embddingTensor.Data(), inputTensor.Data(), ret.Data(), outputShape[len(outputShape)-1])
	return ret, nil
}

func (op *embeddingOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, err
	}

	var (
		emb tensor.Tensor
		in  tensor.Tensor
		ok  bool
	)

	if emb, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("Expected embedding to be a tensor")
	}
	if in, ok = inputs[1].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("Expected input to be a tensor")
	}

	return emb, in, nil
}

func (op *embeddingOp) do(embedding, imput, output interface{}, embSize int) {
	inputV, ok := imput.([]int32)
	if !ok {
		log.Panicf("Expected input to be a []int32, got %T", imput)
	}

	switch v := embedding.(type) {
	case []float32:
		outputV := output.([]float32)
		for i, d := range inputV {
			copy(outputV[i*embSize:(i+1)*embSize], v[int(d)*embSize:int(d+1)*embSize])
		}
	case []float64:
		outputV := output.([]float64)
		for i, d := range inputV {
			copy(outputV[i*embSize:(i+1)*embSize], v[int(d)*embSize:int(d+1)*embSize])
		}
	default:
		log.Panicf("Expected embedding to be a []float32 or []float64, got %T", embedding)
	}
}

func (op embeddingOp) ReturnsPtr() bool {
	return false
}

func (op embeddingOp) CallsExtern() bool {
	return false
}

func (op embeddingOp) OverwritesInput() int {
	return -1
}

func (op embeddingOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op embeddingOp) Hashcode() uint32 {
	return simpleHash(op)
}

func (op embeddingOp) String() string {
	return "Embedding"
}

func (op *embeddingOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 1 {
		return fmt.Errorf("dropout diff requires 1 arguments")
	}

	diff := &dropoutDiffOp{op}
	xdv, ydv := getDV(inputs[0], output)

	_, err := diff.UsePreallocDo(xdv.d, xdv.Value, output.Value(), ydv.d)

	return err
}

func (op *embeddingOp) DiffWRT(inputs int) []bool {
	if inputs != 1 {
		panic(fmt.Sprintf("dropout operator only supports one input, got %d instead", inputs))
	}

	return []bool{true}
}

func (op *embeddingOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	input := inputs[0]
	diff := &dropoutDiffOp{op}

	ret, err := ApplyOp(diff, input, output, grad)
	if err != nil {
		return nil, err
	}

	return Nodes{ret}, nil
}

func (op *embeddingOp) SetTraining(isTraining bool) error { return nil }

type embeddingDiffOp struct {
	*embeddingOp
}

// ensure it complies with the Op interface
var (
	_ Op          = &embeddingOp{}
	_ ADOp        = &embeddingOp{}
	_ SDOp        = &embeddingOp{}
	_ TrainModeOp = &embeddingOp{}
)
