package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"strconv"
	"time"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

const (
	// N is the number of rows in our dataset
	N = 26733
	// feats is the number of features (x) in our dataset
	feats = 10
	// trainIter is the number of interations for which to train
	trainIter = 5000
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write mem profile to file")
var static = flag.Bool("static", false, "Use static test file")
var wT tensor.Tensor
var yT tensor.Tensor
var xT tensor.Tensor

// in this example, we will generate random float64 values
var float = tensor.Float64

// init generates random values for x, w, and y for demo purposes
func init() {
	xBacking := tensor.Random(float, N*feats)
	wBacking := tensor.Random(float, feats)
	var yBacking interface{}
	switch float {
	case tensor.Float64:
		backing := make([]float64, N)
		for i := range backing {
			backing[i] = float64(rand.Intn(2))
		}
		yBacking = backing
	case tensor.Float32:
		backing := make([]float32, N)
		for i := range backing {
			backing[i] = float32(rand.Intn(2))
		}
		yBacking = backing
	}

	xT = tensor.New(tensor.WithBacking(xBacking), tensor.WithShape(N, feats))
	yT = tensor.New(tensor.WithBacking(yBacking), tensor.WithShape(N))
	wT = tensor.New(tensor.WithBacking(wBacking), tensor.WithShape(feats))
}

func main() {
	flag.Parse()
	rand.Seed(1337)
	log.SetFlags(0)

	if *static {
		float = tensor.Float64 // because the loadStatck function only loads []float64
		wBacking, xBacking, yBacking := loadStatic()
		xT = tensor.New(tensor.WithBacking(xBacking), tensor.WithShape(N, feats))
		yT = tensor.New(tensor.WithBacking(yBacking), tensor.WithShape(N))
		wT = tensor.New(tensor.WithBacking(wBacking), tensor.WithShape(feats))
	}

	// To start, we need to create a graph and construct all the nodes.
	// Everything from the input to the prediction needs to be a node.

	// We start by creating nodes for the training
	// create a new graph and add x, y, w, b, and one as Nodes
	g := G.NewGraph()
	x := G.NewMatrix(g, Float, G.WithName("x"), G.WithShape(N, feats))
	y := G.NewVector(g, Float, G.WithName("y"), G.WithShape(N))

	w := G.NewVector(g, Float, G.WithName("w"), G.WithShape(feats))
	b := G.NewScalar(g, Float, G.WithName("bias"))
	// Add a constant node 1 to be used later in the loss function
	one := G.NewConstant(1.0)

	// Here we create the nodes that will do the prediction operations
	// create a node that has the operation: (x*w + b)
	xwmb := G.Must(G.Add(G.Must(G.Mul(x, w)), b))
	// create a node that has the operation: sigmoid(xwmb)
	prob := G.Must(G.Sigmoid(xwmb))
	G.WithName("prob")(prob)
	// create a "pred" node that has the operation (prod > 0.5)
	// this ensures that our prediction output is {0,1}
	pred := G.Must(G.Gt(prob, G.NewConstant(0.5), false))
	G.WithName("pred")(pred)

	// Here we create the nodes that contain the operations that
	// will calculate the cost function.
	// binary cross entropy: -y * log(prob) - (1-y)*log(1-prob)
	logProb := G.Must(G.Log(prob))
	fstTerm := G.Must(G.HadamardProd(G.Must(G.Neg(y)), logProb))
	oneMinusY := G.Must(G.Sub(one, y))
	logOneMinusProb := G.Must(G.Log(G.Must(G.Sub(one, prob))))
	sndTerm := G.Must(G.HadamardProd(oneMinusY, logOneMinusProb))

	crossEntropy := G.Must(G.Sub(fstTerm, sndTerm))
	G.WithName("crossEntropy")(crossEntropy)
	loss := G.Must(G.Mean(crossEntropy))

	// regularization term
	weightSq := G.Must(G.Square(w))
	sumSq := G.Must(G.Sum(weightSq))
	l1reg := G.NewConstant(0.01, G.WithName("l1reg"))
	regTerm := G.Must(G.Mul(l1reg, sumSq))

	// cost we want to minimize
	cost := G.Must(G.Add(loss, regTerm))
	G.WithName("cost")(cost)

	// calculate gradient by backpropagation https://en.wikipedia.org/wiki/Backpropagation
	grads, err := G.Grad(cost, w, b)
	handleError(err)
	// "dcost/dw" == derivative of cost with respect to w
	G.WithName("dcost/dw")(grads[0])
	// "dcost/db" == derivative of cost with respect to b
	G.WithName("dcost/db")(grads[1])

	// create the nodes for calculating the gradient
	learnRate := G.NewConstant(0.1) // be careful not to set a learnRate too high
	gwlr := G.Must(G.Mul(learnRate, grads[0]))
	wUpd := G.Must(G.Sub(w, gwlr))
	gblr := G.Must(G.Mul(learnRate, grads[1]))
	bUpd := G.Must(G.Sub(b, gblr))

	// create the nodes that do the final operation of updating w and b
	G.Set(w, wUpd)
	G.Set(b, bUpd)
	// write to the gographviz file for debugging https://github.com/awalterschulze/gographviz
	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)

	// Now that we have created all the notes, we can compile a program.
	// We are essentially creating a list of instructions to get from our inputs {x, y}
	// to our outputs {wUpd, bUpd}. Note that we only need to create the progam once.
	prog, locMap, err := G.CompileFunction(g, G.Nodes{x, y}, G.Nodes{wUpd, bUpd})
	handleError(err)
	fmt.Printf("%v", prog) // print the instructions
	// With our program, we initialize a new TapeMachine that will execute our program
	machine := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	// we allocated the node w before, but we never set it's initial value
	machine.Let(w, wT)
	machine.Let(b, 0.0)
	// ?? Why do we need to profile the CPU?
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	// Now that we have our graph, program, and machine, we can start training
	start := time.Now()
	for i := 0; i < trainIter; i++ {
		// move the pointer back to the beginning of the prog. Reset() does not delete any values
		machine.Reset()
		// ?? why do we need to reinitialize the values the nodes {x,y}?
		machine.Let(x, xT)
		machine.Let(y, yT)
		handleError(machine.RunAll())
		// ?? Shouldn't the program have already done this? The last node has this Set() operation
		//	G.Set(w, wUpd)
		//	G.Set(b, bUpd)
		machine.Set(w, wUpd)
		machine.Set(b, bUpd)
	}
	fmt.Printf("Time taken: %v\n", time.Since(start))
	fmt.Printf("Final Model: \nw: %3.3s\nb: %+3.3s\n", w.Value(), b.Value())

	fmt.Printf("Target values: %#v\n", yT)
	fmt.Printf("START\n")

	// Now that we have our final model, we need to write a new program
	// that goes from the input {x} to the prediction {pred}
	prog, locMap, err = G.CompileFunction(g, G.Nodes{x}, G.Nodes{pred})
	fmt.Printf("%+v", err)
	handleError(err)
	machine = G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	// ?? Don't we want to use the weight and bias that we trained? not reset them?
	machine.Let(w, wT)
	machine.Let(b, 0.0)
	// ?? Shouldn't we create a new tensor xT so that we can predict?
	machine.Let(x, xT)
	handleError(machine.RunAll())
	fmt.Printf("Predicted: %#v\n", pred.Value())
}
func loadStatic() (w, x, y []float64) {
	d0, err := os.Open("testdata/X_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(d0)
		r.Comma = ','
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			for _, f := range record {
				fl, _ := strconv.ParseFloat(f, 64)
				x = append(x, fl)
			}
		}
		if len(x) != N*feats {
			log.Fatalf("Expected %d*%d. Got %d instead", N, feats, len(x))
		}
	} else {
		log.Println("could not read from file")
		x = tensor.Random(Float, N*feats).([]float64)
	}

	w0, err := os.Open("testdata/W_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(w0)
		r.Comma = ' '
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			fl, _ := strconv.ParseFloat(record[0], 64)
			w = append(w, fl)
		}

		if len(w) != feats {
			log.Fatalf("Expected %d rows. Got %d instead", feats, len(w))
		}
	} else {
		w = tensor.Random(Float, feats).([]float64)
	}

	y0, err := os.Open("testdata/Y_ds1.10.csv")
	if err == nil {
		r := csv.NewReader(y0)
		r.Comma = ','
		for {
			record, err := r.Read()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatal(err)
			}

			fl, _ := strconv.ParseFloat(record[0], 64)
			y = append(y, fl)
		}
		if len(y) != N {
			log.Fatalf("Expected %d rows. Got %d instead", N, len(y))
		}
	} else {
		y = make([]float64, N)
		for i := range y {
			y[i] = float64(rand.Intn(2))
		}

	}
	return
}

func handleError(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}
