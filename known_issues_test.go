package gorgonia

import (
	"testing"

	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestIssue182(t *testing.T) {
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
			t.Fatalf("Machine failed to run at turn %v", turns)
		}
		machine.Reset()
	}

	correct := []float64{-1, -1, -1}
	assert.Equal(correct, dz[0].Value().Data().([]float64))
	if _, ok := a.boundTo.(*dualValue); ok {
		t.Fatalf("Expected constants to not have derivatives")
	}
}

func TestIssue217(t *testing.T) {
	//it works, cost = 22
	if err := issue217(tensor.Shape{2, 2}, tensor.Shape{2, 2}); err != nil {
		t.Fatal(err)
	}

	//panic: Node Σ[0](%2) :: float32, has 0 dimensions(Shape: ()). Input shape is (1, 1)...
	if err := issue217(tensor.Shape{2, 2}, tensor.Shape{2, 1}); err != nil {
		t.Fatal(err)
	}

	//panic: Node Σ[1](%2) :: float32, has 0 dimensions(Shape: ()). Input shape is (1, 1)...
	if err := issue217(tensor.Shape{1, 2}, tensor.Shape{2, 2}); err != nil {
		t.Fatal(err)
	}
}

func issue217(xS, yS tensor.Shape) error {

	g := NewGraph()
	x := NewMatrix(g, Float32, WithName("x"), WithShape(xS...), WithInit(RangedFrom(0)))
	y := NewMatrix(g, Float32, WithName("y"), WithShape(yS...), WithInit(RangedFrom(0)))

	z := Must(Mul(x, y))
	cost := Must(Sum(z))
	//cost := Must(Mean(z))

	_, err := Grad(cost, x, y)
	if err != nil {
		return errors.Wrap(err, "Grad")
	}

	m := NewTapeMachine(g)
	if err = m.RunAll(); err != nil {
		return errors.Wrap(err, "Run")
	}
	return nil
}
