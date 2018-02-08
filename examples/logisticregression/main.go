package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
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
	trainIter = 500
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write mem profile to file")
var static = flag.Bool("static", false, "Use static test file")
var wT tensor.Tensor
var yT tensor.Tensor
var xT tensor.Tensor

// in this example, we will generate random float64 values
var Float = tensor.Float64

// init generates random values for x, w, and y for demo purposes
// Use the static flag to load your own data
func init() {
	xBacking := tensor.Random(Float, N*feats)
	wBacking := tensor.Random(Float, feats)
	var yBacking interface{}
	switch Float {
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
		Float = tensor.Float64 // because the loadStatck function only loads []float64
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
	// create a "pred" node that has the operation that checks if prob is 
	// greater than 0.5. This ensures that our prediction output returns 
	// {true, false}
	pred := G.Must(G.Gt(prob, G.NewConstant(0.5), false))
	G.WithName("pred")(pred)

	// Gorgonia might delete values from nodes so we are going to save it
	// and print it out later
	var predicted G.Value
	readNode := G.Read(pred, &predicted)

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
	G.WithName("loss")(loss)

	// In order to prevent overfitting, we add a L2 regularization term
	weightSq := G.Must(G.Square(w))
	sumSq := G.Must(G.Sum(weightSq))
	l2reg := G.NewConstant(0.01, G.WithName("l2reg"))
	regTerm := G.Must(G.Mul(l2reg, sumSq))

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

	// Run the following line to write to the gographviz file for debugging https://github.com/awalterschulze/gographviz
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)

	// Now that we have created all the notes, we should compile a training program.
	// We are essentially creating a list of instructions to get from our inputs {x, y}
	// to our outputs {wUpd, bUpd, readNode}. Note that we need to tell gorgonia that
	// readNode is one of our outputs so that we can access it.
	prog, locMap, err := G.CompileFunction(g, G.Nodes{x, y}, G.Nodes{wUpd, bUpd, readNode})
	handleError(err)
	fmt.Printf("%v", prog) // print the instructions
	// With our program, we initialize a new TapeMachine that will execute our program
	machine := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))
	// Note that NewTapeMachine() will compile if WithPrecomiled() is not provided.
	// Internally, Gorgonia will figure out that a compilation process needs to happen, 
	// so it will call Compile(g), which will output the prog and locMap internally.
	// When nodes are added to the graph to Gorgonia, nodes that have no precedents 
	// and have a nil Op are marked as input nodes. So Gorgonia knows to start there.
	// But since our graph contains the training and prediction nodes, we manually compiled.

	// we allocated the node w before, but we never set it's initial value
	machine.Let(w, wT)
	machine.Let(b, 0.0)

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
		// We should reinitialize the values {x,y}. This is a good practice.
		// Think of the machine as a function(x,y) and we are providing the input values
		machine.Let(x, xT)
		machine.Let(y, yT)
		handleError(machine.RunAll())
		// After running the machine, we want to update w and b
		machine.Set(w, wUpd)
		machine.Set(b, bUpd)
		
		// After each iteration, we print out the training accuracy to see 
		// how our algorithm is doing
		accuracy := accuracy(y.Value(), predicted)
		fmt.Printf("Interation #%v, Training accuracy: %#v\n", i, accuracy)

		}
	fmt.Printf("Time taken: %v\n", time.Since(start))
	fmt.Printf("Final Model: \nw: %3.3s\nb: %+3.3s\n", w.Value(), b.Value())

	fmt.Printf("START\n")

	// Now that we have our final model, we need to write a new program
	// that goes from the input {x} to the prediction {pred}
	prog, locMap, err = G.CompileFunction(g, G.Nodes{x}, G.Nodes{pred})
	handleError(err)
	machine = G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	machine.Let(w, wT)
	machine.Let(b, 0.0)	
	machine.Let(x, xT)
	handleError(machine.RunAll())
	handleError(err)

}
func accuracy(target, predicted G.Value) (float64){
	count := 0.0
	targetArray := target.Data().([]float64)
	predictedArray := predicted.Data().([]bool)
	targetBool := false
	for i:=0;i<target.Size();i++{
		if targetArray[i] == 1.0{
			targetBool = true
		}else{
			targetBool = false
		}
		if targetBool == predictedArray[i]{
			count++
		}
	}
	return count/float64(target.Size())
}
func handleError(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}