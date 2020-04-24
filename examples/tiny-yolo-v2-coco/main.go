package main

import (
	"fmt"
	"log"
	"time"

	"gorgonia.org/gorgonia"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	width    = 416
	height   = 416
	channels = 3
)

func main() {
	// Init Graph
	g := G.NewGraph()

	model := NewTinyYOLOv2Net(g, 80, 5, "model/yolov2-tiny.weights")

	imgf32, err := GetFloat32Image("data/dog_416x416.jpg")
	if err != nil {
		log.Fatalln(err)
	}

	image := tensor.New(tensor.WithShape(1, channels, height, width), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
	x := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(1, channels, width, height), gorgonia.WithName("x"))

	gorgonia.Let(x, image)
	if err := model.FeedForward(g, x); err != nil {
		log.Fatalf("%+v", err)
	}

	tm := G.NewTapeMachine(g)
	defer tm.Close()
	st := time.Now()
	if err := tm.RunAll(); err != nil {
		log.Fatalf("%+v", err)
	}
	fmt.Println("Feedforwarded in:", time.Since(st))

	dets, err := model.ProcessOutput()
	if err != nil {
		log.Fatalf("%+v", err)
	}
	fmt.Println("Detections:")
	for i := range dets {
		log.Println(dets[i])
	}

	tm.Reset()
}
