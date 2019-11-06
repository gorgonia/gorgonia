package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	f, err := os.Open("theta.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var thetaT *tensor.Dense
	err = dec.Decode(&thetaT)
	if err != nil {
		log.Fatal(err)
	}
	g := gorgonia.NewGraph()
	theta := gorgonia.NodeFromAny(g, thetaT, gorgonia.WithName("theta"))
	values := make([]float64, 5)
	xT := tensor.New(tensor.WithBacking(values))
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	y, err := gorgonia.Mul(x, theta)
	machine := gorgonia.NewTapeMachine(g)
	values[4] = 1.0
	for {
		values[0] = getInput("sepal length")
		values[1] = getInput("sepal widt")
		values[2] = getInput("petal length")
		values[3] = getInput("petal width")

		if err = machine.RunAll(); err != nil {
			log.Fatal(err)
		}
		switch math.Round(y.Value().Data().(float64)) {
		case 1:
			fmt.Println("It is probably a setosa")
		case 2:
			fmt.Println("It is probably a virginica")
		case 3:
			fmt.Println("It is probably a versicolor")
		default:
			fmt.Println("unknown iris")
		}
		machine.Reset()
	}
}

func getInput(s string) float64 {
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("%v: ", s)
	text, _ := reader.ReadString('\n')
	text = strings.Replace(text, "\n", "", -1)

	input, err := strconv.ParseFloat(text, 64)
	if err != nil {
		log.Fatal(err)
	}
	return input
}
