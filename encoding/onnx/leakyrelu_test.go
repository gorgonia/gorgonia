package gorgonnx

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestLeakyRELU_float32(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-1, 0, 1, -2, 3, 4}),
	)

	expectedT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-0.1, 0, 1, -0.2, 3, 4}),
	)

	g := gorgonia.NewGraph()
	xN := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	leakyrelu := &leakyRELU{
		alpha: 0.1,
	}

	output, err := gorgonia.ApplyOp(leakyrelu, xN)
	if err != nil {
		t.Fatal(err)
	}

	m := gorgonia.NewTapeMachine(g)
	err = m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	if len(output.Shape()) != len(expectedT.Shape()) {
		t.Fatalf("wrong dimension, got %v, expect %v", output.Shape(), expectedT.Shape())
	}
	for i := range output.Shape() {
		if output.Shape()[i] != expectedT.Shape()[i] {
			t.Fatalf("wrong dimension, got %v, expect %v", output.Shape(), expectedT.Shape())
		}
	}
	assert.InDeltaSlice(t, output.Value().Data(), expectedT.Data(), 1e-6)
}
func TestLeakyRELU_float64(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float64{-1, 0, 1, -2, 3, 4}),
	)

	expectedT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float64{-0.1, 0, 1, -0.2, 3, 4}),
	)

	g := gorgonia.NewGraph()
	xN := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	leakyrelu := &leakyRELU{
		alpha: 0.1,
	}

	output, err := gorgonia.ApplyOp(leakyrelu, xN)
	if err != nil {
		t.Fatal(err)
	}

	m := gorgonia.NewTapeMachine(g)
	err = m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	if len(output.Shape()) != len(expectedT.Shape()) {
		t.Fatalf("wrong dimension, got %v, expect %v", output.Shape(), expectedT.Shape())
	}
	for i := range output.Shape() {
		if output.Shape()[i] != expectedT.Shape()[i] {
			t.Fatalf("wrong dimension, got %v, expect %v", output.Shape(), expectedT.Shape())
		}
	}
	assert.InDeltaSlice(t, output.Value().Data(), expectedT.Data(), 1e-6)
}
func TestLeakyRELU_int(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]int{-1, 0, 1, -2, 3, 4}),
	)

	g := gorgonia.NewGraph()
	xN := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	leakyrelu := &leakyRELU{
		alpha: 0.1,
	}

	_, err := gorgonia.ApplyOp(leakyrelu, xN)
	if err != nil {
		t.Fatal(err)
	}

	m := gorgonia.NewTapeMachine(g)
	err = m.RunAll()
	if err == nil {
		t.Fatal("Type is not supported, should have triggered an error")
	}
}
