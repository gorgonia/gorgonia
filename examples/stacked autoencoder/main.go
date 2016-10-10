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

func loadMNIST(t string) (inputs, targets types.Tensor) {
	fmt.Println("Loading Training Data..")

	var labelData []Label
	var imageData []RawImage
	var height, width int
	var err error

	var labelFile, dataFile string
	switch t {
	case "train", "dev":
		labelFile = "train-labels.idx1-ubyte"
		dataFile = "train-images.idx3-ubyte"
	case "test":
		labelFile = "t10k-labels.idx1-ubyte"
		dataFile = "t10k-images.idx3-ubyte"

	}

	if labelData, err = readLabelFile(open(labelFile)); err != nil {
		log.Fatal(err)
	}

	if height, width, imageData, err = readImageFile(open(dataFile)); err != nil {
		log.Fatal(err)
	}

	inputs = prepareX(imageData) // transform into floats
	targets = prepareY(labelData)

	if t == "dev" {
		inputs, _ = tensor.Slice(inputs, T.S(0, 1000))
		targets, _ = tensor.Slice(targets, T.S(0, 1000))
	}

	log.Printf("%s Images: %d. | Width: %d, Height: %d\n", t, len(imageData), width, height)
	log.Printf("%s Labels: %d. ", t, len(labelData))
	log.Printf("Inputs: %+s", inputs)
	log.Printf("targets: %+s", targets)
	return inputs, targets
}

func predictTen(logprobs types.Tensor) (guesses []int, err error) {
	argmax, err := tensor.Argmax(logprobs, 1)
	if err != nil {
		return nil, err
	}
	guesses = argmax.Data().([]int)
	return
}

func main() {
	flag.Parse()
	rand.Seed(1337)

	/* EXAMPLE TIME */
	inputs, targets := loadMNIST("dev")

	size := inputs.Shape()[0]
	inputSize := 784
	outputSize := 10
	hiddenSizes := []int{1000, 1000, 1000}
	layers := len(hiddenSizes)
	corruptions := []float64{0.1, 0.2, 0.3}
	batchSize := 1
	pretrainEpoch := 20
	finetuneEpoch := 20

	g := T.NewGraph()
	sda := NewStackedDA(g, batchSize, size, inputSize, outputSize, layers, hiddenSizes, corruptions)
	var err error

	// start CPU profiling before we start training
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	for i := 0; i < pretrainEpoch; i++ {
		if err = sda.Pretrain(inputs, i); err != nil {
			ioutil.WriteFile("fullGraph_err.dot", []byte(g.ToDot()), 0644)
			log.Fatalf("i: %d err :%v", i, err)
		}
	}

	ys := make([]int, targets.Shape()[0])
	ys = ys[:0]
	for i := 0; i < targets.Shape()[0]; i++ {
		ysl, _ := tensor.Slice(targets, T.S(i))
		raw := ysl.Data().([]float64)
		for i, v := range raw {
			if v == 0.9 {
				ys = append(ys, i)
				break
			}
		}
	}
	fmt.Printf("ys: %v\n", ys[0:100])

	log.Printf("Starting to finetune now")
	for i := 0; i < finetuneEpoch; i++ {
		if err = sda.Finetune(inputs, ys, i); err != nil {
			log.Fatal(err)
		}
	}

	// Visualize
	var visualizeLayer int
	log.Printf("Visualizing %dth layer", visualizeLayer)
	finalWeights := sda.autoencoders[0].w.Value().(T.Tensor).Tensor.(*tf64.Tensor).Clone()
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

	/* PREDICTION TIME */

	// here I'm using the test dataset as prediction.
	// in real life you should probably be doing crossvalidations and whatnots
	// but in this demo, we're going to skip all those
	log.Println("pred")
	testX, testY := loadMNIST("dev")
	var one, correct, lp types.Tensor
	if one, err = tensor.Slice(testX, T.S(0, batchSize)); err != nil {
		log.Fatal(err)
	}

	if correct, err = tensor.Slice(testY, T.S(0, batchSize)); err != nil {
		log.Fatal(err)
	}

	var predictions []int
	if lp, err = sda.Forwards(one); err != nil {
		log.Fatal(err)
	}

	if predictions, err = predictTen(lp); err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Correct: \n%+#3.3s. \nPredicted: %v. \nLogprobs: \n%+#3.3s", correct, predictions, lp)
}
