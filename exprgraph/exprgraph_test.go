package exprgraph

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

// TestConstruction tests that the constructions follow several invariances
func TestConstruction(t *testing.T) {
	assert := assert.New(t)

	// usual use case
	g := New(tensor.StdEng{})
	x := Make(g, "x", tensor.WithBacking([]float32{1, 2, 3, 4}), tensor.WithShape(2, 2))
	assert.Equal(1, len(g.nodes))
	id := g.Insert(x)
	assert.Equal(1, len(g.nodes))
	assert.Equal(id, x.NodeID)

	// when we insert the tensor with data back into the graph we expect the same ID
	rawX := x.Tensor.(tensor.Tensor)
	id2 := g.Insert(rawX)
	assert.Equal(id, id2)
	assert.Equal(1, len(g.nodes))

	// but when we create a new raw tensor to insert, we will expect a different ID
	rawX2 := tensor.New(tensor.WithBacking([]float32{1, 2, 3, 4}), tensor.WithShape(2, 2), tensor.WithEngine(g), WithName("x"))
	id3 := g.Insert(rawX2)
	assert.NotEqual(id, id3)
	assert.Equal(2, len(g.nodes))
}
