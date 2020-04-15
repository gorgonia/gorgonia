package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestSoftmax_LargeNumbers(t *testing.T) {
	inputT := tensor.New(
		tensor.WithShape(2, 4),
		tensor.WithBacking([]float64{0, 1, 2, 3, 10000, 10001, 10002, 10003}),
	)
	expectedOutput := tensor.New(
		tensor.WithShape(2, 4),
		tensor.WithBacking([]float64{0.032058604, 0.08714432, 0.23688284, 0.6439143, 0.032058604, 0.08714432, 0.23688284, 0.6439143}),
	)
	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)
	g.ApplyOperation(onnx.Operation{
		Name:       "Softmax",
		Attributes: nil,
	}, output)
	s := &stableSoftmax{
		axis: 1,
	}
	s.apply(g, output.(*Node))
	err := g.Run()
	if err != nil {
		t.Fatal(err)
	}
	outputT := output.(*Node).GetTensor()
	assert.InDeltaSlice(t, expectedOutput.Data(), outputT.Data(), 1e-6, "the two tensors should be equal.")
}
