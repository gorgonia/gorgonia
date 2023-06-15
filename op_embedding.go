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
	arg0, arg1 hm.Type
}

func newEmbeddingOp(at, bt hm.Type) *embeddingOp {
	return &embeddingOp{at, bt}
}

func (op *embeddingOp) Arity() int {
	return 2
}

func (op *embeddingOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	var a0, a1, retType hm.Type
	switch arg0 := op.arg0.(type) {
	case TensorType:
		a0 = makeFromTensorType(arg0, a)
	default:
		a0 = a
	}

	switch arg1 := op.arg1.(type) {
	case TensorType:
		a1 = makeFromTensorType(arg1, a)
		retType = makeTensorType(arg1.Dims+1, a)
	default:
		a1 = a
	}

	return hm.NewFnType(a0, a1, retType)
}

func (op *embeddingOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("embedding requires 2 inputs, got %d", len(inputs))
	}
	weightShape := inputs[0].(tensor.Shape)
	if weightShape.Dims() != 2 {
		return nil, fmt.Errorf("embedding requires a 2D weight matrix, got %dD", weightShape.Dims())
	}
	indicesShape := inputs[1].(tensor.Shape)
	var outputShape []int
	outputShape = append(outputShape, indicesShape...)
	outputShape = append(outputShape, weightShape[1])
	return outputShape, nil
}

func (op *embeddingOp) Do(inputs ...Value) (Value, error) {
	weightTensor, indicesTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("can't check embedding input: %w", err)
	}
	outputShape, err := op.InferShape(weightTensor.Shape(), indicesTensor.Shape())
	if err != nil {
		return nil, fmt.Errorf("can't infer embedding output shape: %w", err)
	}
	ret := tensor.New(tensor.WithShape(outputShape...), tensor.Of(weightTensor.Dtype()))
	op.do(weightTensor.Data(), indicesTensor.Data(), ret.Data(), outputShape[len(outputShape)-1])
	return ret, nil
}

func (op *embeddingOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	weightTensor, indicesTensor, err := op.checkInput(inputs...)
	if err != nil {
		return nil, fmt.Errorf("can't check embedding input: %w", err)
	}
	outputShape, err := op.InferShape(weightTensor.Shape(), indicesTensor.Shape())
	if err != nil {
		return nil, fmt.Errorf("can't infer embedding output shape: %w", err)
	}
	op.do(weightTensor.Data(), indicesTensor.Data(), prealloc.Data(), outputShape[len(outputShape)-1])
	return prealloc, nil
}

func (op *embeddingOp) checkInput(inputs ...Value) (tensor.Tensor, tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, nil, err
	}

	var (
		weight  tensor.Tensor
		indices tensor.Tensor
		ok      bool
	)

	if weight, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("expected weight to be a tensor")
	}
	if indices, ok = inputs[1].(tensor.Tensor); !ok {
		return nil, nil, errors.Errorf("expected indices to be a tensor")
	}

	return weight, indices, nil
}

func (op *embeddingOp) do(weight, indices, output interface{}, embSize int) {
	switch v := weight.(type) {
	case []float32:
		indicesV, ok := indices.([]float32)
		if !ok {
			log.Panicf("expected indices to be a []float32, got %T", indices)
		}
		outputV := output.([]float32)
		for i, d := range indicesV {
			copy(outputV[i*embSize:(i+1)*embSize], v[int(d)*embSize:int(d+1)*embSize])
		}
	case []float64:
		indicesV, ok := indices.([]float64)
		if !ok {
			log.Panicf("expected indices to be a []float64, got %T", indices)
		}
		outputV := output.([]float64)
		for i, d := range indicesV {
			copy(outputV[i*embSize:(i+1)*embSize], v[int(d)*embSize:int(d+1)*embSize])
		}
	default:
		log.Panicf("expected weight to be a []float32 or []float64, got %T", weight)
	}
}

func (op *embeddingOp) ReturnsPtr() bool {
	return false
}

func (op *embeddingOp) CallsExtern() bool {
	return false
}

func (op *embeddingOp) OverwritesInput() int {
	return -1
}

func (op *embeddingOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op *embeddingOp) Hashcode() uint32 {
	return simpleHash(op)
}

func (op *embeddingOp) String() string {
	return "Embedding"
}

func (op *embeddingOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	if len(inputs) != 2 {
		return fmt.Errorf("embedding diff requires 1 arguments")
	}

	diff := &embeddingDiffOp{op}
	wdv := inputs[0].boundTo.(*dualValue)
	xdv := inputs[1].boundTo.(*dualValue)
	ydv := output.boundTo.(*dualValue)

	_, err := diff.UsePreallocDo(wdv.d, wdv.Value, xdv.Value, output.Value(), ydv.d)
	return err
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

	weight := inputs[0]
	indices := inputs[1]
	diff := &embeddingDiffOp{op}

	ret, err := ApplyOp(diff, weight, indices, output, grad)
	if err != nil {
		return nil, err
	}

	return Nodes{ret, nil}, nil
}

func (op *embeddingOp) SetTraining(bool) error { return nil }

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

func (op *embeddingDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return inputs[0].(tensor.Shape), nil
}

func (op *embeddingDiffOp) Do(values ...Value) (Value, error) {
	weight := values[0].(*tensor.Dense)
	indices := values[1].(*tensor.Dense)
	output := values[2].(*tensor.Dense)
	grad := values[3].(*tensor.Dense)

	dy := ZeroValue(weight)
	v, err := op.UsePreallocDo(dy, weight, indices, output, grad)

	return v, err
}

func (op *embeddingDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	weight := inputs[0].(*tensor.Dense)
	indices := inputs[1].(*tensor.Dense)
	outGrad := inputs[3].(*tensor.Dense)
	result := prealloc.(*tensor.Dense)
	embSize := weight.Shape()[1]

	switch weight.Dtype() {
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

func (op *embeddingDiffOp) ReturnsPtr() bool {
	return true
}

func (op *embeddingDiffOp) CallsExtern() bool {
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
