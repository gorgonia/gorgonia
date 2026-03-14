package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestTapeMachineRefresh(t *testing.T) {
	g := NewGraph()

	x := NewVector(g, tensor.Float64, WithShape(2), WithName("x"))
	y := NewVector(g, tensor.Float64, WithShape(2), WithName("y"))

	z := Must(Add(x, y))

	m := NewTapeMachine(g)

	// Bind values
	Let(x, tensor.New(tensor.WithBacking([]float64{1, 2}), tensor.WithShape(2)))
	Let(y, tensor.New(tensor.WithBacking([]float64{3, 4}), tensor.WithShape(2)))

	// Run forward pass
	err := m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	// Check result
	if z.Value() == nil {
		t.Fatal("z should have a value after RunAll")
	}

	// Now compute gradients
	grads, err := Grad(z, x, y)
	if err != nil {
		t.Fatal(err)
	}

	// Run again - should detect new nodes and refresh
	err = m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	// Check that gradients have values
	for i, grad := range grads {
		if grad.Value() == nil {
			t.Fatalf("gradient %d should have a value after refresh", i)
		}
	}
}

func TestTapeMachineGradientExecution(t *testing.T) {
	g := NewGraph()

	x := NewScalar(g, tensor.Float64, WithName("x"))
	y := NewScalar(g, tensor.Float64, WithName("y"))

	z := Must(Mul(x, y))
	loss := Must(Add(z, NewConstant(1.0)))

	m := NewTapeMachine(g)

	// Bind values
	Let(x, 2.0)
	Let(y, 3.0)

	// Run forward
	err := m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	// Compute gradients
	_, err = Grad(loss, x, y)
	if err != nil {
		t.Fatal(err)
	}

	// Run again to execute gradients
	err = m.RunAll()
	if err != nil {
		t.Fatal(err)
	}

	// Verify gradient values
	dx, err := x.Grad()
	if err != nil {
		t.Fatal(err)
	}
	if dx == nil {
		t.Fatal("x gradient should not be nil")
	}

	dy, err := y.Grad()
	if err != nil {
		t.Fatal(err)
	}
	if dy == nil {
		t.Fatal("y gradient should not be nil")
	}

	// Check values (dz/dx = y = 3, dz/dy = x = 2)
	dxVal := dx.Data().(float64)
	if dxVal != 3.0 {
		t.Fatalf("expected dx=3.0, got %v", dxVal)
	}

	dyVal := dy.Data().(float64)
	if dyVal != 2.0 {
		t.Fatalf("expected dy=2.0, got %v", dyVal)
	}
}
