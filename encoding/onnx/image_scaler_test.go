package gorgonnx

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestImageScaler(t *testing.T) {
	xT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-1, 0, 1, -2, 3, 4}),
	)
	expectedT := tensor.New(
		tensor.WithShape(1, 2, 1, 3),
		tensor.WithBacking([]float32{-1*0.1 + 1, 0*0.1 + 1, 1*0.1 + 1, -2*0.1 - 1, 3*0.1 - 1, 4*0.1 - 1}),
	)
	imageScaler := &imageScaler{
		scale: 0.1,
		bias:  []float32{1, -1},
	}
	output, err := imageScaler.Do(xT)
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
	assert.InDeltaSlice(t, output.Data(), expectedT.Data(), 1e-6)

}
