package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestMinBetween(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	a := NodeFromAny(g, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{
		1000, 2,
		3, 4,
	})), WithName("a"))
	b := NodeFromAny(g, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{
		100, 200,
		300, 400,
	})), WithName("b"))

	op := minBetween{}
	c, err := ApplyOp(op, a, b)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Sum(c)
	if err != nil {
		t.Fatal(err)
	}

	grads, err := Grad(s, a, b)
	if err != nil {
		t.Fatal(err)
	}

	m := NewTapeMachine(g, TraceExec())
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correctGradA := []float64{0, 1, 1, 1}
	correctGradB := []float64{1, 0, 0, 0}

	assert.Equal(correctGradA, grads[0].Value().Data())
	assert.Equal(correctGradB, grads[1].Value().Data())
}

func TestMaxBetween(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	a := NodeFromAny(g, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{
		1000, 2,
		3, 4,
	})), WithName("a"))
	b := NodeFromAny(g, tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{
		100, 200,
		300, 400,
	})), WithName("b"))

	op := maxBetween{}
	c, err := ApplyOp(op, a, b)
	if err != nil {
		t.Fatal(err)
	}
	s, err := Sum(c)
	if err != nil {
		t.Fatal(err)
	}

	grads, err := Grad(s, a, b)
	if err != nil {
		t.Fatal(err)
	}

	m := NewTapeMachine(g, TraceExec())
	if err = m.RunAll(); err != nil {
		t.Fatal(err)
	}

	correctGradA := []float64{1, 0, 0, 0}
	correctGradB := []float64{0, 1, 1, 1}

	assert.Equal(correctGradA, grads[0].Value().Data())
	assert.Equal(correctGradB, grads[1].Value().Data())
}
