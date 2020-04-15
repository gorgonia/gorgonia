package gorgonnx

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestGraph_badnode(t *testing.T) {
	inputT := tensor.New(
		tensor.WithShape(2, 4),
		tensor.WithBacking([]float32{0, 1, 2, 3, 10000, 10001, 10002, 10003}),
	)
	g := NewGraph()
	input := g.NewNode()
	g.AddNode(input)
	output := g.NewNode()
	g.AddNode(output)
	g.SetWeightedEdge(g.NewWeightedEdge(output, input, 0))
	input.(*Node).SetTensor(inputT)
	s := &stableSoftmax{
		axis: 1,
	}
	s.apply(g, output.(*Node))
	err := g.Run()
	if err == nil {
		t.Fatal("should raise an error because output is not a tensor nor an operation")
	}
}
