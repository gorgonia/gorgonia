package gorgonia_test

import (
	"testing"

	"gorgonia.org/gorgonia"
)

func TestMean_issue375(t *testing.T) {
	g := gorgonia.NewGraph()

	w0 := gorgonia.NewTensor(g, gorgonia.Float32, 4, gorgonia.WithShape(1, 64, 1, 64), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	result, _ := gorgonia.Mean(w0, 3)
	t.Log(result)
}
