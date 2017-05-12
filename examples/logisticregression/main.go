package main

// import "C"

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"runtime/pprof"
	"time"

	G "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	// cblas "github.com/gonum/blas/cgo"
)

const (
	// N     = 400
	// feats = 784

	N     = 26733
	feats = 10

	trainIter = 5000
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")
var memprofile = flag.String("memprofile", "", "write mem profile to file")
var static = flag.Bool("static", false, "Use static test file")
var wT tensor.Tensor
var yT tensor.Tensor
var xT tensor.Tensor

var Float = tensor.Float64

func handleError(err error) {
	if err != nil {
		log.Fatalf("%+v", err)
	}
}

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

	// G.Use(cblas.Implementation{})
	// G.UseNonStable()

	g := G.NewGraph()
	x := G.NewMatrix(g, Float, G.WithName("x"), G.WithShape(N, feats))
	y := G.NewVector(g, Float, G.WithName("y"), G.WithShape(N))

	w := G.NewVector(g, Float, G.WithName("w"), G.WithShape(feats))
	b := G.NewScalar(g, Float, G.WithName("bias"))

	one := G.NewConstant(1.0)

	// prob = 1/(1 + e^(-(x·w)-b))
	xwmb := G.Must(G.Add(G.Must(G.Mul(x, w)), b))
	prob := G.Must(G.Sigmoid(xwmb))
	G.WithName("prob")(prob)

	// "forward"
	pred := G.Must(G.Gt(prob, G.NewConstant(0.5), false))
	G.WithName("pred")(pred)

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

	// calculate gradient
	grads, err := G.Grad(cost, w, b)
	handleError(err)

	G.WithName("dcost/dw")(grads[0])
	G.WithName("dcost/db")(grads[1])

	// gradient updates
	learnRate := G.NewConstant(0.1)
	gwlr := G.Must(G.Mul(learnRate, grads[0]))
	wUpd := G.Must(G.Sub(w, gwlr))
	gblr := G.Must(G.Mul(learnRate, grads[1]))
	bUpd := G.Must(G.Sub(b, gblr))

	G.Set(w, wUpd)
	G.Set(b, bUpd)

	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)

	prog, locMap, err := G.CompileFunction(g, G.Nodes{x, y}, G.Nodes{wUpd, bUpd})
	handleError(err)
	fmt.Printf("%v", prog)
	machine := G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

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

	start := time.Now()
	for i := 0; i < trainIter; i++ {
		machine.Reset()
		machine.Let(x, xT)
		machine.Let(y, yT)
		handleError(machine.RunAll())

		machine.Set(w, wUpd)
		machine.Set(b, bUpd)
	}
	fmt.Printf("Time taken: %v\n", time.Since(start))
	fmt.Printf("Final Model: \nw: %3.3s\nb: %+3.3s\n", w.Value(), b.Value())

	fmt.Printf("Target values: %#v\n", yT)
	prog, locMap, err = G.CompileFunction(g, G.Nodes{x}, G.Nodes{pred})
	handleError(err)
	machine = G.NewTapeMachine(g, G.WithPrecompiled(prog, locMap))

	machine.Let(w, wT)
	machine.Let(b, 0.0)
	machine.Let(x, xT)
	handleError(machine.RunAll())
	fmt.Printf("Predicted: %#v\n", pred.Value())
}
