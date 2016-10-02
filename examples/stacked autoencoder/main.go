package main

import (
	"fmt"
	"io/ioutil"
	"log"

	T "github.com/chewxy/gorgonia"
)

func main() {
	fmt.Println("Loading Training Data..")

	labelData, err := readLabelFile(open("train-labels-idx1-ubyte"))
	height, width, imageData, err2 := readImageFile(open("train-images-idx3-ubyte"))

	if err != nil || err2 != nil {
		log.Fatalf("Err: %v | Err2: %v", err, err2)
	}

	fmt.Printf("Images: %d. | Width: %d, Height: %d\n", len(imageData), width, height)
	fmt.Printf("Labels: %d. ", len(labelData))

	inputs := prepareX(imageData) // transform into floats
	targets := prepareY(labelData)
	fmt.Printf("inputs: %+s\n", inputs)
	fmt.Printf("targets: %+s\n", targets)

	/* EXAMPLE TIME */

	xV, err := inputs.Slice(s(0))
	if err != nil {
		log.Println(err)
	}
	fmt.Printf("%+3.3s", xV)

	g := T.NewGraph()

	x := T.NewNodeFromAny(g, xV, T.WithName("x"))

	size := inputs.Shape()[0]
	inputSize := 784
	outputSize := 10
	layers := 1
	hiddenSizes := []int{784}
	corruptions := []float64{0.1}
	sda := NewStackedDA(g, size, inputSize, outputSize, layers, hiddenSizes, corruptions)

	sda.Pretrain(x)
	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
}
