// 286 is a program to test issue 286

package main

import (
	"flag"
	"log"
	"math/rand"

	_ "net/http/pprof"

	"github.com/pkg/errors"
	"gorgonia"
	"gorgonia.org/tensor"
)

var (
	epochs     = flag.Int("epochs", 10, "Number of epochs to train for")
	dataset    = flag.String("dataset", "train", "Which dataset to train on? Valid options are \"train\" or \"test\"")
	dtype      = flag.String("dtype", "float64", "Which dtype to use")
	batchsize  = flag.Int("batchsize", 10, "Batch size")
	cpuprofile = flag.String("cpuprofile", "", "CPU profiling")
)

const loc = "./mnist/"

var dt tensor.Dtype

func parseDtype() {
	switch *dtype {
	case "float64":
		dt = tensor.Float64
	case "float32":
		dt = tensor.Float32
	default:
		log.Fatalf("Unknown dtype: %v", *dtype)
	}
}

type nn struct {
	g      *gorgonia.ExprGraph
	w0, w1 *gorgonia.Node

	out     *gorgonia.Node
	predVal gorgonia.Value
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }

func newNN(g *gorgonia.ExprGraph) *nn {
	// Create node for w/weight
	w0 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(784, 300), gorgonia.WithName("w0"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w1 := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(300, 10), gorgonia.WithName("w1"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1}
}

func (m *nn) fwd(x *gorgonia.Node) (err error) {
	var l0, l1 *gorgonia.Node
	var l0dot *gorgonia.Node

	// Set first layer to be copy of input
	l0 = x

	// Dot product of l0 and w0, use as input for ReLU
	if l0dot, err = gorgonia.Mul(l0, m.w0); err != nil {
		return errors.Wrap(err, "Unable to multiply l0 and w0")
	}

	// l0dot := gorgonia.Must(gorgonia.Mul(l0, m.w0))

	// Build hidden layer out of result
	l1 = gorgonia.Must(gorgonia.Rectify(l0dot))

	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l1, m.w1); err != nil {
		return errors.Wrapf(err, "Unable to multiply l1 and w1")
	}

	m.out, err = gorgonia.SoftMax(out)
	gorgonia.Read(m.out, &m.predVal)
	return

}

func main() {
	flag.Parse()
	parseDtype()
	rand.Seed(7945)

	var err error

	bs := *batchsize
	g := gorgonia.NewGraph()
	x := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 784), gorgonia.WithName("x"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	m := newNN(g)
	if err = m.fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}

	losses, err := gorgonia.HadamardProd(m.out, y)
	if err != nil {
		log.Fatal(err)
	}
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))

	// we wanna track costs
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)

	if _, err = gorgonia.Grad(cost, m.learnables()...); err != nil {
		log.Fatal(err)
	}

	vm := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(m.learnables()...))
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))

	if err = vm.RunAll(); err != nil {
		log.Fatalf("Failed %v", err)
	}

	solver.Step(gorgonia.NodesToValueGrads(m.learnables()))
	vm.Reset()
}
