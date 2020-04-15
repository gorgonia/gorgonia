package gorgonnx_test

import (
	"testing"

	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

func TestRun_multiple(t *testing.T) {
	simpleAdd := []byte{0x8, 0x3, 0x12, 0xc, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65, 0x73, 0x74, 0x3a, 0x69, 0xa, 0x10, 0xa, 0x1, 0x78, 0xa, 0x1, 0x79, 0x12, 0x3, 0x73, 0x75, 0x6d, 0x22, 0x3, 0x41, 0x64, 0x64, 0x12, 0x8, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x61, 0x64, 0x64, 0x5a, 0x17, 0xa, 0x1, 0x78, 0x12, 0x12, 0xa, 0x10, 0x8, 0x1, 0x12, 0xc, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x4, 0xa, 0x2, 0x8, 0x5, 0x5a, 0x17, 0xa, 0x1, 0x79, 0x12, 0x12, 0xa, 0x10, 0x8, 0x1, 0x12, 0xc, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x4, 0xa, 0x2, 0x8, 0x5, 0x62, 0x19, 0xa, 0x3, 0x73, 0x75, 0x6d, 0x12, 0x12, 0xa, 0x10, 0x8, 0x1, 0x12, 0xc, 0xa, 0x2, 0x8, 0x3, 0xa, 0x2, 0x8, 0x4, 0xa, 0x2, 0x8, 0x5, 0x42, 0x2, 0x10, 0x9}
	inputA0 := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}),
	)
	inputA1 := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
	)
	inputB1 := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
	)
	output1 := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}),
	)
	output2 := tensor.New(
		tensor.WithShape(3, 4, 5),
		tensor.WithBacking([]float32{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}),
	)
	backend := gorgonnx.NewGraph()
	model := onnx.NewModel(backend)
	err := model.UnmarshalBinary(simpleAdd)
	if err != nil {
		t.Fatal(err)
	}
	model.SetInput(0, inputA1)
	model.SetInput(1, inputB1)
	err = backend.Run()
	// Check error
	output, err := model.GetOutputTensors()
	if err != nil {
		t.Fatal(err)
	}
	if !equal(output2.Data().([]float32), output[0].Data().([]float32)) {
		t.Fatal("operation failed at first pass, results differs")
	}
	model.SetInput(0, inputA0)
	model.SetInput(1, inputB1)
	err = backend.Run()
	// Check error
	output, err = model.GetOutputTensors()
	if err != nil {
		t.Fatal(err)
	}
	if !equal(output1.Data().([]float32), output[0].Data().([]float32)) {
		t.Fatalf("operation failed at second pass, results differs (%v and %v)", output1.Data().([]float32)[0], output[0].Data().([]float32)[0])
	}
}

func equal(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
