package engine

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConv2d_F32(t *testing.T) {
	g := NewGraph()
	xV := tensor.New(tensor.WithShape(1, 1, 5, 5), tensor.WithBacking([]float32{
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
	}))
	kernelV := tensor.New(tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float32{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}))

	x := g.NewTensor(Float32, 4, WithShape(1, 1, 5, 5), WithValue(xV), WithName("x"))
	g.AddNode(x)
	w := g.NewTensor(Float32, 4, WithShape(1, 1, 3, 3), WithValue(kernelV), WithName("w"))
	g.AddNode(w)
	z := g.NewNode().(*Node)
	g.AddNode(z)
	g.SetWeightedEdge(g.NewWeightedEdge(z, x, 1.0))
	g.SetWeightedEdge(g.NewWeightedEdge(z, w, 2.0))

	err := g.ApplyOp(NewConv2d(tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}), z)
	if err != nil {
		t.Fatal(err)
	}
	// logger := log.New(os.Stderr, "", 0)
	// vm := NewTapeMachine(g, WithLogger(logger), WithWatchlist(), WithValueFmt("%#v"))
	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}

	correct := []float32{
		2, 3, 3, 3, 2,
		6, 9, 9, 9, 6,
		12, 18, 18, 18, 12,
		18, 27, 27, 27, 18,
		14, 21, 21, 21, 14,
	}
	t.Logf("%v", z.Value())

	assert.Equal(t, correct, z.Value().Data())
}

func TestConv2d_F64(t *testing.T) {
	g := NewGraph()
	xV := tensor.New(tensor.WithShape(1, 1, 5, 5), tensor.WithBacking([]float64{
		0, 0, 0, 0, 0,
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
	}))
	kernelV := tensor.New(tensor.WithShape(1, 1, 3, 3), tensor.WithBacking([]float64{
		1, 1, 1,
		1, 1, 1,
		1, 1, 1,
	}))

	x := g.NewTensor(Float64, 4, WithShape(1, 1, 5, 5), WithValue(xV), WithName("x"))
	g.AddNode(x)
	w := g.NewTensor(Float64, 4, WithShape(1, 1, 3, 3), WithValue(kernelV), WithName("w"))
	g.AddNode(w)
	z := g.NewNode().(*Node)
	g.AddNode(z)
	g.SetWeightedEdge(g.NewWeightedEdge(z, x, 1.0))
	g.SetWeightedEdge(g.NewWeightedEdge(z, w, 2.0))

	err := g.ApplyOp(NewConv2d(tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1}), z)
	if err != nil {
		t.Fatal(err)
	}
	// logger := log.New(os.Stderr, "", 0)
	// vm := NewTapeMachine(g, WithLogger(logger), WithWatchlist(), WithValueFmt("%#v"))
	vm := NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		t.Fatal(err)
	}

	correct := []float64{
		2, 3, 3, 3, 2,
		6, 9, 9, 9, 6,
		12, 18, 18, 18, 12,
		18, 27, 27, 27, 18,
		14, 21, 21, 21, 14,
	}

	assert.Equal(t, correct, z.Value().Data())

}
