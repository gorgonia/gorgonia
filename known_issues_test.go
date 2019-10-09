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
	if err != nil {
		t.Fatal(err)
	}
	z, err := Mul(y, b)
	if err != nil {
		t.Fatal(err)
	}
	dz, err := Grad(z, x)
	if err != nil {
		t.Fatal(err)
	}
	machine := NewTapeMachine(g)
	defer machine.Close()

	machine.Let(x, xT)
	machine.Let(b, -0.5)
	for turns := 0; turns < 4; turns++ {
		if err := machine.RunAll(); err != nil {
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

func TestIssue268_im2col(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, tensor.Float32, 4, WithShape(1, 2, 5, 5), WithInit(RangedFrom(0)))
	yT := tensor.New(tensor.WithShape(1, 5, 5, 18), tensor.WithBacking([]float32{
		0, 0, 0, 0, 0, 1, 0, 5, 6, 0, 0, 0, 0, 25, 26, 0, 30, 31, 0, 0, 0, 0, 1, 2, 5, 6, 7, 0, 0, 0, 25, 26, 27, 30,
		31, 32, 0, 0, 0, 1, 2, 3, 6, 7, 8, 0, 0, 0, 26, 27, 28, 31, 32, 33, 0, 0, 0, 2, 3, 4, 7, 8, 9, 0, 0, 0, 27, 28,
		29, 32, 33, 34, 0, 0, 0, 3, 4, 0, 8, 9, 0, 0, 0, 0, 28, 29, 0, 33, 34, 0, 0, 0, 1, 0, 5, 6, 0, 10, 11, 0, 25,
		26, 0, 30, 31, 0, 35, 36, 0, 1, 2, 5, 6, 7, 10, 11, 12, 25, 26, 27, 30, 31, 32, 35, 36, 37, 1, 2, 3, 6, 7, 8,
		11, 12, 13, 26, 27, 28, 31, 32, 33, 36, 37, 38, 2, 3, 4, 7, 8, 9, 12, 13, 14, 27, 28, 29, 32, 33, 34, 37, 38,
		39, 3, 4, 0, 8, 9, 0, 13, 14, 0, 28, 29, 0, 33, 34, 0, 38, 39, 0, 0, 5, 6, 0, 10, 11, 0, 15, 16, 0, 30, 31, 0, 35,
		36, 0, 40, 41, 5, 6, 7, 10, 11, 12, 15, 16, 17, 30, 31, 32, 35, 36, 37, 40, 41, 42, 6, 7, 8, 11, 12, 13, 16,
		17, 18, 31, 32, 33, 36, 37, 38, 41, 42, 43, 7, 8, 9, 12, 13, 14, 17, 18, 19, 32, 33, 34, 37, 38, 39, 42, 43,
		44, 8, 9, 0, 13, 14, 0, 18, 19, 0, 33, 34, 0, 38, 39, 0, 43, 44, 0, 0, 10, 11, 0, 15, 16, 0, 20, 21, 0, 35, 36,
		0, 40, 41, 0, 45, 46, 10, 11, 12, 15, 16, 17, 20, 21, 22, 35, 36, 37, 40, 41, 42, 45, 46, 47, 11, 12, 13, 16,
		17, 18, 21, 22, 23, 36, 37, 38, 41, 42, 43, 46, 47, 48, 12, 13, 14, 17, 18, 19, 22, 23, 24, 37, 38, 39, 42,
		43, 44, 47, 48, 49, 13, 14, 0, 18, 19, 0, 23, 24, 0, 38, 39, 0, 43, 44, 0, 48, 49, 0, 0, 15, 16, 0, 20, 21, 0,
		0, 0, 0, 40, 41, 0, 45, 46, 0, 0, 0, 15, 16, 17, 20, 21, 22, 0, 0, 0, 40, 41, 42, 45, 46, 47, 0, 0, 0, 16, 17,
		18, 21, 22, 23, 0, 0, 0, 41, 42, 43, 46, 47, 48, 0, 0, 0, 17, 18, 19, 22, 23, 24, 0, 0, 0, 42, 43, 44, 47, 48,
		49, 0, 0, 0, 18, 19, 0, 23, 24, 0, 0, 0, 0, 43, 44, 0, 48, 49, 0, 0, 0, 0,
	}))
	y, err := Im2Col(x, []int{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		t.Fatal(err)
	}

	machine := NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(t, yT.Data(), y.Value().Data(), 1e-5, "Tensors should be the same")
}

func TestIssue273_maxpool_pads(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, tensor.Float32, 4, WithShape(1, 2, 5, 5), WithInit(RangedFrom(0)))
	yT := tensor.New(
		tensor.WithShape(1, 2, 7, 7),
		tensor.WithBacking([]float32{
			0, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 9, 9, 9, 10, 11, 12, 13, 14, 14, 14, 15, 16,
			17, 18, 19, 19, 19, 20, 21, 22, 23, 24, 24, 24, 20, 21, 22, 23, 24, 24, 24,
			20, 21, 22, 23, 24, 24, 24, 25, 26, 27, 28, 29, 29, 29, 30, 31, 32, 33, 34,
			34, 34, 35, 36, 37, 38, 39, 39, 39, 40, 41, 42, 43, 44, 44, 44, 45, 46, 47,
			48, 49, 49, 49, 45, 46, 47, 48, 49, 49, 49, 45, 46, 47, 48, 49, 49, 49,
		}))

	y, err := MaxPool2D(x, []int{3, 3}, []int{2, 2}, []int{1, 1})
	if err != nil {
		t.Fatal(err)
	}

	machine := NewTapeMachine(g)
	if err = machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	assert.Equal(t, yT.Shape(), y.Shape(), "Tensors should be the same")
	assert.InDeltaSlice(t, yT.Data(), y.Value().Data(), 1e-5, "Tensors should be the same")

}

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
