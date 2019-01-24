package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestBroadcast_BCHW(t *testing.T) {
	assert := assert.New(t)

	g := NewGraph()

	x := g.NodeFromAny(tensor.New(
		tensor.WithShape(1, 2, 3, 3),
		tensor.WithBacking([]float32{
			0, 1, 2,
			3, 4, 5,
			5, 6, 7,
			8, 9, 10,
			11, 12, 13,
			14, 15, 16,
		})),
		WithName("x"))
	g.AddNode(x)

	y := g.NodeFromAny(tensor.New(
		tensor.WithShape(1, 2, 1, 1),
		tensor.WithBacking([]float32{100, 100})),
		WithName("y"))
	g.AddNode(y)
	/*
		reshapedY := g.NewNode().(*Node)
		g.AddNode(reshapedY)
		g.SetWeightedEdge(g.NewWeightedEdge(reshapedY, y, 1.0))
		reshapeOp := NewReshapeOperation(tensor.Shape([]int{1, 2, 1, 1}))
		err := g.ApplyOp(reshapeOp, reshapedY)
		if err != nil {
			t.Fatal(err)
		}
	*/
	sum := g.NewNode().(*Node)
	sum.name = "sum"
	g.AddNode(sum)
	g.SetWeightedEdge(g.NewWeightedEdge(sum, x, 0.0))
	//g.SetWeightedEdge(g.NewWeightedEdge(sum, reshapedY, 1.0))
	g.SetWeightedEdge(g.NewWeightedEdge(sum, y, 1.0))
	op := NewAddOperation(nil, []byte{0, 2, 3})

	err := g.ApplyOp(op, sum)
	if err != nil {
		t.Fatal(err)
	}
	/*
		gviz, err := dot.Marshal(g)
		if err != nil {
			t.Fatal(err)
		}
		fmt.Println(string(gviz))
	*/

	// logger := log.New(os.Stderr, "", 0)
	// machine := NewTapeMachine(g, WithLogger(logger))
	machine := NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	sumT := tensor.New(
		tensor.WithShape(1, 2, 3, 3),
		tensor.WithBacking([]float32{
			100, 101, 102,
			103, 104, 105,
			105, 106, 107,
			108, 109, 110,
			111, 112, 113,
			114, 115, 116,
		}))

	assert.Equal(sumT.Shape(), sum.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(sumT.Data(), sum.Value().Data(), 1e-5, "Tensors should be the same")

}
