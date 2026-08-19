package tflite_test

import (
	"math/rand"
	"testing"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/gorgonia/encoding/tflite"
	"gorgonia.org/tensor"
)

// checkModel asserts that buf looks like a TFLite flatbuffer.
func checkModel(t *testing.T, buf []byte) {
	t.Helper()
	if len(buf) < 8 {
		t.Fatalf("model too short: %d bytes", len(buf))
	}
	if string(buf[4:8]) != "TFL3" {
		t.Fatalf("bad file identifier: %q", buf[4:8])
	}
}

func TestWriter(t *testing.T) {
	m := tflite.NewModel()
	x := m.Input("x", []int{1, 3})
	f := m.Constant("w", []int{2, 3}, []float32{1, 2, 3, -1, 0.5, 0})
	b := m.Constant("b", []int{2}, []float32{0.5, -0.5})
	h := m.Tanh(m.FullyConnected(x, f, b, tflite.ActNone))
	m.Output(m.Softmax(h, 1.0))
	buf, err := m.Bytes()
	if err != nil {
		t.Fatal(err)
	}
	checkModel(t, buf)
}

func TestWriterErrors(t *testing.T) {
	m := tflite.NewModel()
	if _, err := m.Bytes(); err == nil {
		t.Fatal("expected an error for a model without outputs")
	}

	m = tflite.NewModel()
	m.Output(m.Constant("w", []int{2, 3}, []float32{1}))
	if _, err := m.Bytes(); err == nil {
		t.Fatal("expected an error for a bad constant")
	}
}

func randTensor(r *rand.Rand, shape ...int) *tensor.Dense {
	n := 1
	for _, s := range shape {
		n *= s
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	return tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
}

func TestExportMLP(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 4), G.WithName("x"))
	w1 := G.NewMatrix(g, tensor.Float32, G.WithShape(4, 8), G.WithName("w1"), G.WithValue(randTensor(r, 4, 8)))
	b1 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 8), G.WithName("b1"), G.WithValue(randTensor(r, 1, 8)))
	w2 := G.NewMatrix(g, tensor.Float32, G.WithShape(8, 3), G.WithName("w2"), G.WithValue(randTensor(r, 8, 3)))
	b2 := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 3), G.WithName("b2"), G.WithValue(randTensor(r, 1, 3)))

	h := G.Must(G.Tanh(G.Must(G.Add(G.Must(G.Mul(x, w1)), b1))))
	y := G.Must(G.SoftMax(G.Must(G.Add(G.Must(G.Mul(h, w2)), b2))))

	buf, err := tflite.Export(y, x)
	if err != nil {
		t.Fatal(err)
	}
	checkModel(t, buf)
}

func TestExportSigmoid(t *testing.T) {
	r := rand.New(rand.NewSource(2))

	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 2), G.WithName("x"))
	w := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 2), G.WithName("w"), G.WithValue(randTensor(r, 2, 2)))

	y := G.Must(G.Sigmoid(G.Must(G.Mul(x, w))))

	buf, err := tflite.Export(y, x)
	if err != nil {
		t.Fatal(err)
	}
	checkModel(t, buf)
}

func TestExportUnsupported(t *testing.T) {
	g := G.NewGraph()
	x := G.NewMatrix(g, tensor.Float32, G.WithShape(1, 2), G.WithName("x"))
	y := G.Must(G.Square(x))
	if _, err := tflite.Export(y, x); err == nil {
		t.Fatal("expected an error for an unsupported op")
	}

	// A weight leaf without a bound value must be reported.
	g = G.NewGraph()
	x = G.NewMatrix(g, tensor.Float32, G.WithShape(1, 2), G.WithName("x"))
	w := G.NewMatrix(g, tensor.Float32, G.WithShape(2, 2), G.WithName("w"))
	y = G.Must(G.Mul(x, w))
	if _, err := tflite.Export(y, x); err == nil {
		t.Fatal("expected an error for a weight without a value")
	}
}
