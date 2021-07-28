package stdops

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

func TestAddition(t *testing.T) {
	var a, b values.Value
	a = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6}))
	b = tensor.New(tensor.WithShape(2, 3), tensor.WithBacking([]float64{10, 20, 30, 40, 50, 60}))

	op := Add{}
	c, err := op.Do(context.Background(), a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work")
	}
	correct := []float64{11, 22, 33, 44, 55, 66}
	assert.Equal(t, correct, c.Data())
	correctShape := tensor.Shape{2, 3}
	assert.True(t, correctShape.Eq(c.Shape()))

	a = tensor.New(tensor.WithShape(), tensor.WithBacking([]int{1}))
	b = tensor.New(tensor.WithShape(), tensor.WithBacking([]int{2}))
	c = tensor.New(tensor.WithShape(), tensor.WithBacking([]int{-1}))
	c, err = op.PreallocDo(context.Background(), c, a, b)
	if err != nil {
		t.Fatalf("Expected addition operation to work")
	}
	correctScalar := 3
	assert.Equal(t, correctScalar, c.Data())
	correctShape = tensor.ScalarShape()
	assert.True(t, correctShape.Eq(c.Shape()))
}
