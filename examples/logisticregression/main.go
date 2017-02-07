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

	T "github.com/chewxy/gorgonia"
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

	// T.Use(cblas.Implementation{})
	// T.UseNonStable()

	g := T.NewGraph()
	x := T.NewMatrix(g, Float, T.WithName("x"), T.WithShape(N, feats))
	y := T.NewVector(g, Float, T.WithName("y"), T.WithShape(N))

	w := T.NewVector(g, Float, T.WithName("w"), T.WithShape(feats))
	b := T.NewScalar(g, Float, T.WithName("bias"))

	one := T.NewConstant(1.0)

	// prob = 1/(1 + e^(-(x·w)-b))
	xwmb := T.Must(T.Add(T.Must(T.Mul(x, w)), b))
	prob := T.Must(T.Sigmoid(xwmb))
	T.WithName("prob")(prob)

	// "forward"
	pred := T.Must(T.Gt(prob, T.NewConstant(0.5), false))
	T.WithName("pred")(pred)

	// binary cross entropy: -y * log(prob) - (1-y)*log(1-prob)
	logProb := T.Must(T.Log(prob))
	fstTerm := T.Must(T.HadamardProd(T.Must(T.Neg(y)), logProb))

	oneMinusY := T.Must(T.Sub(one, y))
	logOneMinusProb := T.Must(T.Log(T.Must(T.Sub(one, prob))))
	sndTerm := T.Must(T.HadamardProd(oneMinusY, logOneMinusProb))

	crossEntropy := T.Must(T.Sub(fstTerm, sndTerm))
	T.WithName("crossEntropy")(crossEntropy)

	loss := T.Must(T.Mean(crossEntropy))

	// regularization term
	weightSq := T.Must(T.Square(w))
	sumSq := T.Must(T.Sum(weightSq))
	l1reg := T.NewConstant(0.01, T.WithName("l1reg"))
	regTerm := T.Must(T.Mul(l1reg, sumSq))

	// cost we want to minimize
	cost := T.Must(T.Add(loss, regTerm))
	T.WithName("cost")(cost)

	// calculate gradient
	grads, err := T.Grad(cost, w, b)
	handleError(err)

	T.WithName("dcost/dw")(grads[0])
	T.WithName("dcost/db")(grads[1])

	// gradient updates
	learnRate := T.NewConstant(0.1)
	gwlr := T.Must(T.Mul(learnRate, grads[0]))
	wUpd := T.Must(T.Sub(w, gwlr))
	gblr := T.Must(T.Mul(learnRate, grads[1]))
	bUpd := T.Must(T.Sub(b, gblr))

	T.Set(w, wUpd)
	T.Set(b, bUpd)

	ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)

	prog, locMap, err := T.CompileFunction(g, T.Nodes{x, y}, T.Nodes{wUpd, bUpd})
	handleError(err)
	fmt.Printf("%v", prog)
	machine := T.NewTapeMachine(prog, locMap)

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
	prog, locMap, err = T.CompileFunction(g, T.Nodes{x}, T.Nodes{pred})
	handleError(err)
	machine = T.NewTapeMachine(prog, locMap)

	machine.Let(w, wT)
	machine.Let(b, 0.0)
	machine.Let(x, xT)
	handleError(machine.RunAll())
	fmt.Printf("Predicted: %#v\n", pred.Value())
}
