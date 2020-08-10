package main

import (
	"fmt"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	imgWidth       = 416
	imgHeight      = 416
	channels       = 3
	boxes          = 3
	leakyCoef      = 0.1
	weights        = "./data/yolov3-tiny.weights"
	cfg            = "./data/yolov3-tiny.cfg"
	cocoClasses    = []string{"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}
	scoreThreshold = float32(0.8)
	iouThreshold   = float32(0.3)
)

func main() {
	g := G.NewGraph()

	input := G.NewTensor(g, tensor.Float32, 4, G.WithShape(1, channels, imgWidth, imgHeight), G.WithName("input"))
	model, err := NewYoloV3Tiny(g, input, len(cocoClasses), boxes, leakyCoef, cfg, weights)
	if err != nil {
		fmt.Printf("Can't prepare YOLOv3 network due the error: %s\n", err.Error())
		return
	}
	model.Print()

	imgf32, err := GetFloat32Image("data/dog_416x416.jpg", imgHeight, imgWidth)
	if err != nil {
		fmt.Printf("Can't read []float32 from image due the error: %s\n", err.Error())
		return
	}
	image := tensor.New(tensor.WithShape(1, channels, imgHeight, imgWidth), tensor.Of(tensor.Float32), tensor.WithBacking(imgf32))
	err = G.Let(input, image)
	if err != nil {
		fmt.Printf("Can't let input = []float32 due the error: %s\n", err.Error())
		return
	}

	tm := G.NewTapeMachine(g)
	defer tm.Close()

	// Feedforward
	st := time.Now()
	if err := tm.RunAll(); err != nil {
		fmt.Printf("Can't run tape machine due the error: %s\n", err.Error())
		return
	}
	fmt.Println("Feedforwarded in:", time.Since(st))

	// Postprocessing
	st = time.Now()
	dets, err := model.ProcessOutput(cocoClasses, scoreThreshold, iouThreshold)
	if err != nil {
		fmt.Printf("Can't do postprocessing due error: %s", err.Error())
		return
	}
	fmt.Println("Postprocessed in:", time.Since(st))

	fmt.Println("Detections:")
	for i := range dets {
		fmt.Println(dets[i])
	}

	tm.Reset()
}
