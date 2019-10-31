package main

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://www.kaggle.com/amarpandey/implementing-linear-regression-on-iris-dataset/notebook
//
func main() {
	xT, yT := getXY()
	g := gorgonia.NewGraph()
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	y := gorgonia.NodeFromAny(g, yT, gorgonia.WithName("y"))
	theta := gorgonia.NewVector(
		g,
		gorgonia.Float64,
		gorgonia.WithName("theta"),
		gorgonia.WithShape(xT.Shape()[1]),
		gorgonia.WithInit(gorgonia.Gaussian(0, 1)))

	pred := must(gorgonia.Mul(x, theta))

	// Gorgonia might delete values from nodes so we are going to save it
	// and print it out later
	var predicted gorgonia.Value
	gorgonia.Read(pred, &predicted)

	squaredError := must(gorgonia.Square(must(gorgonia.Sub(pred, y))))
	cost := must(gorgonia.Mean(squaredError))

	if _, err := gorgonia.Grad(cost, theta); err != nil {
		log.Fatalf("Failed to backpropagate: %v", err)
	}

	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(theta))
	defer machine.Close()

	model := []gorgonia.ValueGrad{theta}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.001))

	fa := mat.Formatted(getThetaNormal(), mat.Prefix("   "), mat.Squeeze())

	fmt.Printf("ϴ: %v\n", fa)
	iter := 10000
	var err error
	for i := 0; i < iter; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
		}

		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("theta: %2.2f  Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
			theta.Value(),
			i,
			cost.Value(),
			accuracy(predicted.Data().([]float64), y.Value().Data().([]float64)))

		machine.Reset() // Reset is necessary in a loop like this
	}
	fmt.Println("")
	err = save(theta.Value())
	if err != nil {
		log.Fatal(err)
	}

}

func accuracy(prediction, y []float64) float64 {
	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}
	return ok / float64(len(y))
}

func getXYMat() (*mat.Dense, *mat.Dense) {
	f, err := os.Open("iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	df := dataframe.ReadCSV(f)
	xDF := df.Drop("species")

	toValue := func(s series.Series) series.Series {
		records := s.Records()
		floats := make([]float64, len(records))
		for i, r := range records {
			switch r {
			case "setosa":
				floats[i] = 1
			case "virginica":
				floats[i] = 2
			case "versicolor":
				floats[i] = 3
			default:
				log.Fatalf("unknown iris: %v\n", r)
			}
		}
		return series.Floats(floats)
	}

	yDF := df.Select("species").Capply(toValue)
	numRows, _ := xDF.Dims()
	xDF = xDF.Mutate(series.New(one(numRows), series.Float, "bias"))
	fmt.Println(xDF.Describe())
	fmt.Println(yDF.Describe())

	return mat.DenseCopyOf(&matrix{xDF}), mat.DenseCopyOf(&matrix{yDF})
}

func getXY() (*tensor.Dense, *tensor.Dense) {
	x, y := getXYMat()

	xT := tensor.FromMat64(x)
	yT := tensor.FromMat64(y)
	// Get rid of the last dimension to create a vector
	s := yT.Shape()
	yT.Reshape(s[0])
	return xT, yT
}

func getThetaNormal() *mat.Dense {
	x, y := getXYMat()
	xt := mat.DenseCopyOf(x).T()
	var xtx mat.Dense
	xtx.Mul(xt, x)
	var invxtx mat.Dense
	invxtx.Inverse(&xtx)
	var xty mat.Dense
	xty.Mul(xt, y)
	var output mat.Dense
	output.Mul(&invxtx, &xty)

	return &output
}

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}

func must(n *gorgonia.Node, err error) *gorgonia.Node {
	if err != nil {
		panic(err)
	}
	return n
}

func one(size int) []float64 {
	one := make([]float64, size)
	for i := 0; i < size; i++ {
		one[i] = 1.0
	}
	return one
}

func save(value gorgonia.Value) error {
	f, err := os.Create("theta.bin")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	err = enc.Encode(value)
	if err != nil {
		return err
	}
	return nil
}
