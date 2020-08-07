package main

import (
	"fmt"
	"runtime"
	"testing"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func BenchmarkSample(b *testing.B) {

	g := G.NewGraph()

	input := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, imgWidth, imgHeight), gorgonia.WithName("input"))
	model, err := NewYoloV3Tiny(g, input, len(cocoClasses), boxes, leakyCoef, cfg, weights)
	if err != nil {
		fmt.Printf("Can't prepare YOLOv3 network due the error: %s\n", err.Error())
		return
	}
	_ = model
	imgf32, err := GetFloat32Image("data/dog_416x416.jpg", imgHeight, imgWidth)
	if err != nil {
		fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
		return
	}
	image := tensor.New(tensor.WithShape(1, channels, imgHeight, imgWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
	err = gorgonia.Let(input, image)
	if err != nil {
		fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
		return
	}

	tm := G.NewTapeMachine(g)
	defer tm.Close()
	for i := 0; i < b.N; i++ {
		if err := tm.RunAll(); err != nil {
			fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
			return
		}
		tm.Reset()
	}

	runtime.GC()
}
