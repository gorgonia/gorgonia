package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestConstDeriv(t *testing.T) {
	// This test revolves around repeated calls to run a VM.
	// Formerly, upon running the VM once, the derivation of the constant is set.
	// This derivation value would get Add()ed to upon subsequqent calls to run the VM.
	//
	// This behaviour was fixed to make sure constants do not have derivatives
	assert := assert.New(t)

	// Build the graph
	g := NewGraph()

	aback := []float64{2.0, 2.0, 2.0}
	x := NewVector(g, tensor.Float64, WithName("x"), WithShape(3))
	a := NewConstant(tensor.New(tensor.WithBacking(aback), tensor.WithShape(3)))

	b := NewScalar(g, tensor.Float64)

	xT := tensor.New(tensor.WithBacking([]float64{1, 1, 1}), tensor.WithShape(3))
	y, err := Mul(x, a)
	z, err := Mul(y, b)
	dz, err := Grad(z, x)
	machine := NewTapeMachine(g)
	defer machine.Close()

	machine.Let(x, xT)
	machine.Let(b, -0.5)
	for turns := 0; turns < 4; turns++ {
		if err = machine.RunAll(); err != nil {
			t.Fatalf("Machine failed to run at turn %v", turn)
		}
		machine.Reset()
	}

	correct := []float64{-1, -1, -1}
	assert.Equal(correct, dz[0].Value().Data().([]float64))
	if _, ok := a.boundTo.(*dualValue); ok {
		t.Fatalf("Expected constants to not have derivatives")
	}
}
