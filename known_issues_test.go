package gorgonia

import (
	"testing"

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

// func TestIssue217(t *testing.T) {
// 	//it works, cost = 22
// 	if err := issue217(tensor.Shape{2, 2}, tensor.Shape{2, 2}); err != nil {
// 		t.Fatal(err)
// 	}

// 	//panic: Node Σ[0](%2) :: float32, has 0 dimensions(Shape: ()). Input shape is (1, 1)...
// 	if err := issue217(tensor.Shape{2, 2}, tensor.Shape{2, 1}); err != nil {
// 		t.Fatal(err)
// 	}

// 	//panic: Node Σ[1](%2) :: float32, has 0 dimensions(Shape: ()). Input shape is (1, 1)...
// 	if err := issue217(tensor.Shape{1, 2}, tensor.Shape{2, 2}); err != nil {
// 		t.Fatal(err)
// 	}
// }

// func issue217(xS, yS tensor.Shape) error {

// 	g := NewGraph()
// 	x := NewMatrix(g, Float32, WithName("x"), WithShape(xS...), WithInit(RangedFrom(0)))
// 	y := NewMatrix(g, Float32, WithName("y"), WithShape(yS...), WithInit(RangedFrom(0)))

// 	z := Must(Mul(x, y))
// 	cost := Must(Sum(z))
// 	//cost := Must(Mean(z))

// 	_, err := Grad(cost, x, y)
// 	if err != nil {
// 		return errors.Wrap(err, "Grad")
// 	}

// 	m := NewTapeMachine(g)
// 	if err = m.RunAll(); err != nil {
// 		return errors.Wrap(err, "Run")
// 	}
// 	return nil
// }

func TestIssue233_F32(t *testing.T) {
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

	x := NewTensor(g, Float32, 4, WithShape(1, 1, 5, 5), WithValue(xV), WithName("x"))
	w := NewTensor(g, Float32, 4, WithShape(1, 1, 3, 3), WithValue(kernelV), WithName("w"))

	y, err := Conv2d(x, w, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
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
	t.Logf("%v", y.Value())

	assert.Equal(t, correct, y.Value().Data())
}

func TestIssue233_F64(t *testing.T) {
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

	x := NewTensor(g, Float64, 4, WithShape(1, 1, 5, 5), WithValue(xV), WithName("x"))
	w := NewTensor(g, Float64, 4, WithShape(1, 1, 3, 3), WithValue(kernelV), WithName("w"))

	y, err := Conv2d(x, w, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
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

	assert.Equal(t, correct, y.Value().Data())

}
