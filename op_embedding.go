package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"
	"log"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type embeddingOp struct {
}

func (op *embeddingOp) Arity() int {
	return 2
}

func (op *embeddingOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a, a)
}

func (op *embeddingOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("embedding requires 2 inputs, got %d", len(inputs))
	}
	embeddingShape := inputs[0].(tensor.Shape)
	if embeddingShape.Dims() != 2 {
		return nil, fmt.Errorf("embedding requires a 2D embedding matrix, got %dD", embeddingShape.Dims())
	}
	inputShape := inputs[1].(tensor.Shape)
	outputShape := []int{}
	outputShape = append(outputShape, inputShape...)
	outputShape = append(outputShape, embeddingShape[1])
	return outputShape, nil
}

func (op *embeddingOp) Do(value ...Value) (Value, error) {
	embddingTensor, inputTensor, err := op.checkInput(value...)
	if err != nil {
		return nil, fmt.Errorf("can't check Embedding input: %w", err)
	}
	outputShape, err := op.InferShape(embddingTensor.Shape(), inputTensor.Shape())
	if err != nil {
		return nil, fmt.Errorf("can't infer Embedding output shape: %w", err)
	}
	ret := tensor.New(tensor.WithShape(outputShape...), tensor.Of(embddingTensor.Dtype()))
	op.do(embddingTensor.Data(), inputTensor.Data(), ret.Data(), outputShape[len(outputShape)-1])
	return ret, nil
}

func (op *embeddingOp) UsePreallocDo(prealloc Value, value ...Value) (Value, error) {
	embddingTensor, inputTensor, err := op.checkInput(value...)
	if err != nil {
		return nil, fmt.Errorf("can't check Embedding input: %w", err)
	}
	outputShape, err := op.InferShape(embddingTensor.Shape(), inputTensor.Shape())
	if err != nil {
		return nil, fmt.Errorf("can't infer Embedding output shape: %w", err)
	}
	op.do(embddingTensor.Data(), inputTensor.Data(), prealloc.Data(), outputShape[len(outputShape)-1])
	return prealloc, nil
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
	switch v := embedding.(type) {
	case []float32:
		inputV, ok := imput.([]float32)
		if !ok {
			log.Panicf("Expected input to be a []int32, got %T", imput)
		}
		outputV := output.([]float32)
		for i, d := range inputV {
			copy(outputV[i*embSize:(i+1)*embSize], v[int(d)*embSize:int(d+1)*embSize])
		}
	case []float64:
		inputV, ok := imput.([]float64)
		if !ok {
			log.Panicf("Expected input to be a []int32, got %T", imput)
		}
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

	// diff := &dropoutDiffOp{op}
	// xdv, ydv := getDV(inputs[0], output)

	// _, err := diff.UsePreallocDo(xdv.d, xdv.Value, output.Value(), ydv.d)

	// return err
	return nil
}

func (op *embeddingOp) DiffWRT(inputs int) []bool {
	if inputs != 2 {
		panic(fmt.Sprintf("embedding operator only supports two input, got %d instead", inputs))
	}

	return []bool{true, false}
}

func (op *embeddingOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	weights := inputs[0]
	indices := inputs[1]
	diff := &embeddingDiffOp{op}

	ret, err := ApplyOp(diff, weights, indices, output, grad)
	if err != nil {
		return nil, err
	}

	return Nodes{ret, nil}, nil
}

func (op *embeddingOp) SetTraining(isTraining bool) error { return nil }

type embeddingDiffOp struct {
	*embeddingOp
}

func (op *embeddingDiffOp) Arity() int {
	return 4
}

func (op *embeddingDiffOp) Type() hm.Type {
	t := hm.TypeVariable('a')
	return hm.NewFnType(t, t, t, t, t)
}

func (op *embeddingDiffOp) InferShape(ds ...DimSizer) (tensor.Shape, error) {
	return ds[0].(tensor.Shape), nil
}

func (op *embeddingDiffOp) Do(values ...Value) (Value, error) {
	weights := values[0].(*tensor.Dense)
	indices := values[1].(*tensor.Dense)
	output := values[2].(*tensor.Dense)
	grad := values[3].(*tensor.Dense)

	dy := ZeroValue(weights)
	v, err := op.UsePreallocDo(dy, weights, indices, output, grad)

	return v, err
}

func (op *embeddingDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	weights := inputs[0].(*tensor.Dense)
	indices := inputs[1].(*tensor.Dense)
	outGrad := inputs[3].(*tensor.Dense)
	result := prealloc.(*tensor.Dense)
	embSize := weights.Shape()[1]

	switch weights.Dtype() {
	case Float64:
		indicesA := indices.Float64s()
		dy := outGrad.Float64s()
		dx := result.Float64s()

		for i, index := range indicesA {
			for j := 0; j < embSize; j++ {
				dx[int(index)*embSize+j] = dy[i*embSize+j]
			}
		}
	case Float32:
		indicesA := indices.Float32s()
		dy := outGrad.Float32s()
		dx := result.Float32s()

		for i, index := range indicesA {
			for j := 0; j < embSize; j++ {
				dx[int(index)*embSize+j] = dy[i*embSize+j]
			}
		}
	}

	return prealloc, nil
}

func (op embeddingDiffOp) ReturnsPtr() bool {
	return true
}

func (op embeddingDiffOp) CallsExtern() bool {
	return false
}

func (op *embeddingDiffOp) OverwritesInput() int {
	return -1
}

func (op *embeddingDiffOp) WriteHash(h hash.Hash) {
	h.Write([]byte(op.String()))
}

func (op *embeddingDiffOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *embeddingDiffOp) String() string {
	return "embeddingDiffOp"
}

var (
	_ Op          = &embeddingOp{}
	_ ADOp        = &embeddingOp{}
	_ SDOp        = &embeddingOp{}
	_ TrainModeOp = &embeddingOp{}
)
