package main

import (
	"flag"
	"fmt"
	"image/jpeg"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"

	T "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write memory profile to this file")

func loadMNIST() (inputs, targets types.Tensor) {
	fmt.Println("Loading Training Data..")

	labelData, err := readLabelFile(open("train-labels.idx1-ubyte"))
	height, width, imageData, err2 := readImageFile(open("train-images.idx3-ubyte"))

	if err != nil || err2 != nil {
		log.Fatalf("Err: %v | Err2: %v", err, err2)
	}

	fmt.Printf("Images: %d. | Width: %d, Height: %d\n", len(imageData), width, height)
	fmt.Printf("Labels: %d. ", len(labelData))

	inputs = prepareX(imageData) // transform into floats
	targets = prepareY(labelData)
	fmt.Printf("inputs: %+s\n", inputs)
	fmt.Printf("targets: %+s\n", targets)

	return inputs, targets
}

func loadShared() (inputs, targets types.Tensor) {
	fx, err := os.OpenFile("shared/x.npy", os.O_RDONLY, 0644)
	fy, err := os.OpenFile("shared/y.npy", os.O_RDONLY, 0644)

	xV := new(tf64.Tensor)
	yV := new(tf64.Tensor)

	err = xV.ReadNpy(fx)
	if err != nil {
		panic(err)
	}
	err = yV.ReadNpy(fy)
	if err != nil {
		panic(err)
	}
	inputs = xV
	targets = yV

	return
}

func main() {
	flag.Parse()
	rand.Seed(1337)

	/* EXAMPLE TIME */
	inputs, targets := loadMNIST()
	// inputs, targets := loadShared()

	xV, err := tensor.Slice(inputs, nil)
	if err != nil {
		log.Println(err)
	}
	fmt.Printf("%+3.3s\n", xV)

	g := T.NewGraph()

	T.UseNonStable()

	size := inputs.Shape()[0]
	inputSize := 784
	outputSize := 10
	hiddenSizes := []int{1000}
	layers := len(hiddenSizes)
	corruptions := []float64{0.1, 0.2, 0.3}
	batchSize := 100
	sda := NewStackedDA(g, batchSize, size, inputSize, outputSize, layers, hiddenSizes, corruptions)
	pretrainEpoch := 5

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	xVD := xV.Data().([]float64)
	if hasOne(xVD) {
		log.Fatal("WTF????!!!!")
	}

	for i := 0; i < pretrainEpoch; i++ {
		if err = sda.Pretrain(xV, i); err != nil {
			ioutil.WriteFile("fullGraph_err.dot", []byte(g.ToDot()), 0644)

			if ver, ok := err.(T.Valuer); ok {
				log.Printf("V: %v: %+3.3s", ver, ver.Value())
			}
			log.Fatalf("i: %d err :%v", i, err)
		}

		ioutil.WriteFile("fullGraph_0.dot", []byte(g.ToDot()), 0644)
	}

	yV, _ := tensor.Slice(targets, T.S(0, 10000))
	ys := make([]int, 10000)
	ys = ys[:0]
	for i := 0; i < yV.Shape()[0]; i++ {
		ysl, _ := tensor.Slice(yV, T.S(i))
		raw := ysl.Data().([]float64)
		for i, v := range raw {
			if v == 0.9 {
				ys = append(ys, i)
				break
			}
		}
	}

	// log.Printf("Starting to finetune now")
	// for i := 0; i < 10; i++ {
	// 	log.Printf("Finetune iter: %d", i)
	// 	sda.Finetune(xV, ys)
	// }
	// log.Printf("Writing images now")

	// Visualize
	finalWeights := sda.autoencoders[0].w.Value().(T.Tensor).Tensor.(*tf64.Tensor)
	finalWeights.T()
	finalWeights.Transpose()
	for i := 0; i < finalWeights.Shape()[0]; i++ {
		rowT, _ := finalWeights.Slice(T.S(i))
		row := rowT.Data().([]float64)
		img := visualizeRow(row)

		f, _ := os.OpenFile(fmt.Sprintf("images/%d.jpg", i), os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
		jpeg.Encode(f, img, &jpeg.Options{jpeg.DefaultQuality})
		f.Close()
	}
	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
}
