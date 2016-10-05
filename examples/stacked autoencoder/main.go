package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"log"
	"os"

	T "github.com/chewxy/gorgonia"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
)

func main() {
	fmt.Println("Loading Training Data..")

	labelData, err := readLabelFile(open("train-labels.idx1-ubyte"))
	height, width, imageData, err2 := readImageFile(open("train-images.idx3-ubyte"))

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

	xV, err := inputs.Slice(T.S(0, 10000))
	if err != nil {
		log.Println(err)
	}
	fmt.Printf("%+3.3s\n", xV)

	g := T.NewGraph()

	size := inputs.Shape()[0]
	inputSize := 784
	outputSize := 10
	hiddenSizes := []int{1000, 1000}
	layers := len(hiddenSizes)
	corruptions := []float64{0.1, 0.2}
	batchSize := 100
	sda := NewStackedDA(g, batchSize, size, inputSize, outputSize, layers, hiddenSizes, corruptions)

	log.Printf("SDA: %v %v", sda.autoencoders[0].w.Shape(), sda.autoencoders[0].h.w.Shape())
	/*



	 */

	// log.Printf("xV: %+#0.1s", xV)
	f, _ := os.Create("test.jpg")
	row := xV.Data().([]float64)
	img := visualizeRow(row)
	jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})

	r := image.Rect(0, 0, 28, 28)
	img2 := image.NewGray(r)
	img2.Pix = []byte(imageData[0])
	f2, _ := os.Create("test2.jpg")
	jpeg.Encode(f2, img2, &jpeg.Options{jpeg.DefaultQuality})

	/*

	 */

	log.Printf("sda.autoencoders[0].w: %+1.1s", sda.autoencoders[0].w.Value())
	for i := 0; i < 10; i++ {
		log.Printf("Pretrain: %d", i)
		if err = sda.Pretrain(xV); err != nil {
			ioutil.WriteFile("fullGraph_err.dot", []byte(g.ToDot()), 0644)
			log.Fatalf("%d %v", i, err)
		}

		ioutil.WriteFile("fullGraph_0.dot", []byte(g.ToDot()), 0644)
	}

	yV, _ := targets.Slice(T.S(0, 10000))
	ys := make([]int, 10000)
	ys = ys[:0]
	for i := 0; i < yV.Shape()[0]; i++ {
		ysl, _ := yV.Slice(T.S(i))
		raw := ysl.Data().([]float64)
		for i, v := range raw {
			if v == 0.9 {
				ys = append(ys, i)
				break
			}
		}
	}
	log.Printf("Starting to finetune now")
	for i := 0; i < 10; i++ {
		sda.Finetune(xV, ys)
	}
	log.Printf("Writing images now")

	finalWeights := sda.autoencoders[0].w.Value().(T.Tensor).Tensor.(*tf64.Tensor)
	finalWeights.T()
	finalWeights.Transpose()
	log.Printf("finalWeights: %v | %v", finalWeights.Shape(), finalWeights.Shape()[0])
	for i := 0; i < finalWeights.Shape()[0]; i++ {
		rowT, _ := finalWeights.Slice(T.S(i))
		row := rowT.Data().([]float64)
		img := visualizeRow(row)

		f, _ := os.OpenFile(fmt.Sprintf("images/%d.jpg", i), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
		f.Close()
	}

	log.Printf("xV: %+0.1s", xV)
	log.Printf("sda.autoencoders[0].w: %+1.1s", sda.autoencoders[0].w.Value())
	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
}
