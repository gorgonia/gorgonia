package gorgonnx

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestBatchnorm_check(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(2, 1, 3),
		tensor.WithBacking([]float32{-1, 0, 1, 2, 3, 4}),
	)
	batchNormOp := &fastBatchnorm{
		epsilon: 1e-5,
	}
	_, err := batchNormOp.check(xT)
	if err == nil {
		t.Fatal("should have triggered an error")
	}

	scaleT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{1, 1.5}),
	)

	biasT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{0, 1}),
	)

	meanT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{0, 3}),
	)

	varT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{1, 1.5}),
	)
	batchNormOp = &fastBatchnorm{
		scale:   scaleT,
		bias:    biasT,
		mean:    meanT,
		varN:    varT,
		epsilon: 1e-5,
	}
	xT = tensor.New(
		tensor.WithShape(2, 1, 3),
		tensor.WithBacking([]float32{-1, 0, 1, 2, 3, 4}),
	)

	_, err = batchNormOp.check(xT)
	if err == nil {
		t.Fatal("should have triggered an error")
	}
	xT = tensor.New(
		tensor.WithShape(1, 3, 1, 2),
		tensor.WithBacking([]float32{-1, 0, 1, 2, 3, 4}),
	)

	_, err = batchNormOp.check(xT)
	if err == nil {
		t.Fatal("should have triggered an error")
	}

}
func TestBatchnorm_float32(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-1, 0, 1, 2, 3, 4}),
	)

	scaleT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{1, 1.5}),
	)

	biasT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{0, 1}),
	)

	meanT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{0, 3}),
	)

	varT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float32{1, 1.5}),
	)
	expectedT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-0.999995, 0, 0.999995, -0.22474074, 1, 2.2247407}),
	)

	g := gorgonia.NewGraph()
	xN := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	scaleN := gorgonia.NodeFromAny(g, scaleT, gorgonia.WithName("scale"))
	biasN := gorgonia.NodeFromAny(g, biasT, gorgonia.WithName("bias"))
	meanN := gorgonia.NodeFromAny(g, meanT, gorgonia.WithName("mean"))
	varN := gorgonia.NodeFromAny(g, varT, gorgonia.WithName("var"))
	batchNormOp := &fastBatchnorm{
		scale:   scaleN.Value(),
		bias:    biasN.Value(),
		mean:    meanN.Value(),
		varN:    varN.Value(),
		epsilon: 1e-5,
	}

	output, err := gorgonia.ApplyOp(batchNormOp, xN)
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
func TestBatchnorm_float64(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float64{-1, 0, 1, 2, 3, 4}),
	)

	scaleT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float64{1, 1.5}),
	)

	biasT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float64{0, 1}),
	)

	meanT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float64{0, 3}),
	)

	varT := tensor.New(
		tensor.WithShape(2),
		tensor.WithBacking([]float64{1, 1.5}),
	)
	expectedT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float64{-0.999995, 0, 0.999995, -0.22474074, 1, 2.2247407}),
	)

	g := gorgonia.NewGraph()
	xN := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	scaleN := gorgonia.NodeFromAny(g, scaleT, gorgonia.WithName("scale"))
	biasN := gorgonia.NodeFromAny(g, biasT, gorgonia.WithName("bias"))
	meanN := gorgonia.NodeFromAny(g, meanT, gorgonia.WithName("mean"))
	varN := gorgonia.NodeFromAny(g, varT, gorgonia.WithName("var"))
	batchNormOp := &fastBatchnorm{
		scale:   scaleN.Value(),
		bias:    biasN.Value(),
		mean:    meanN.Value(),
		varN:    varN.Value(),
		epsilon: 1e-5,
	}

	output, err := gorgonia.ApplyOp(batchNormOp, xN)
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
