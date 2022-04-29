package gorgonia

import (
	"log"
	"math/rand"
	"runtime"
	"testing"
	"time"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// testNN3 is a simple 3 layer NN with its input
type testNN3 struct {
	g *ExprGraph

	in, out *Node
	w1, b1  *Node
	w2, b2  *Node
	w3, b3  *Node
}

func (t *testNN3) Graph() *ExprGraph { return t.g }
func (t *testNN3) X() *Node          { return t.in }
func (t *testNN3) Pred() *Node       { return t.out }
func (t *testNN3) Model() Nodes      { return Nodes{t.w1, t.b1, t.w2, t.b2, t.w3, t.b3} }

func randTestNN3(isDet bool) *testNN3 {
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	// dt
	var dt tensor.Dtype
	dtx := rnd.Intn(2)
	switch dtx {
	case 0:
		dt = tensor.Float64
	case 1:
		dt = tensor.Float32
	}

	// set up data : (n, f)
	var n, f int
	// for n = rnd.Intn(100); n == 0; n = rnd.Intn(100) {
	// }
	// for f = rnd.Intn(50); f == 0; f = rnd.Intn(50) {
	// }
	n = 64
	f = 5

	// simple 3 layer NN with I as its activation func
	var w1f, w2f, w3f InitWFn = GlorotU(1), GlorotU(1), GlorotU(1)
	if isDet {
		w1f = detWeights64_5(1)
		w2f = detWeights64_5(2)
		w3f = detWeights64_5(3)
	}

	g := NewGraph()
	x := NewMatrix(g, dt, WithShape(n, f), WithName("x"), WithInit(Zeroes()))
	w1 := NewMatrix(g, dt, WithShape(f, f), WithName("w1"), WithInit(w1f))
	b1 := NewMatrix(g, dt, WithShape(1, f), WithName("b1"), WithInit(Zeroes()))
	w2 := NewMatrix(g, dt, WithShape(f, f), WithName("w2"), WithInit(w2f))
	b2 := NewMatrix(g, dt, WithShape(1, f), WithName("b2"), WithInit(Zeroes()))
	w3 := NewMatrix(g, dt, WithShape(f, 1), WithName("w3"), WithInit(w3f))
	b3 := NewMatrix(g, dt, WithShape(1, 1), WithName("b3"), WithInit(Zeroes()))

	log.Printf("w2 %1.1f", w1.Value().Data())

	xw1 := Must(Mul(x, w1))
	l1 := Must(Auto(BroadcastAdd, xw1, b1))

	l1w2 := Must(Mul(l1, w2))
	l2 := Must(Auto(BroadcastAdd, l1w2, b2))

	l2w3 := Must(Mul(l2, w3))
	l3 := Must(Auto(BroadcastAdd, l2w3, b3))

	return &testNN3{
		g:   g,
		in:  x,
		out: l3,
		w1:  w1,
		b1:  b1,
		w2:  w2,
		b2:  b2,
		w3:  w3,
		b3:  b3,
	}
}

func detWeights64_5(n int) InitWFn {
	return func(dt tensor.Dtype, s ...int) interface{} {
		switch dt {
		case tensor.Float64:
			switch n {
			case 1:
				return []float64{-0.4, 0.6, -0.6, 0.1, -0.3, 0.4, 0.4, 0.6, -0.7, 0.4, 0.6, 0.7, -0.6, -0.7, -0.7, 0.4, -0.2, -0.3, 0.3, 0.1, 0.1, 0.0, 0.4, 0.7, -0.7}
			case 2:
				return []float64{0.4, -0.6, -0.3, -0.4, -0.0, 0.0, -0.1, 0.6, 0.2, -0.1, -0.3, -0.4, -0.5, -0.4, 0.5, 0.1, 0.2, -0.2, 0.3, 0.3, 0.2, -0.8, -0.9, 0.6, 0.4}
			case 3:
				return []float64{0.4, 0.6, 0.6, -0.4, 0.2}
			}

		case tensor.Float32:
			switch n {
			case 1:
				return []float32{-0.4, 0.6, -0.6, 0.1, -0.3, 0.4, 0.4, 0.6, -0.7, 0.4, 0.6, 0.7, -0.6, -0.7, -0.7, 0.4, -0.2, -0.3, 0.3, 0.1, 0.1, 0.0, 0.4, 0.7, -0.7}
			case 2:
				return []float32{0.4, -0.6, -0.3, -0.4, -0.0, 0.0, -0.1, 0.6, 0.2, -0.1, -0.3, -0.4, -0.5, -0.4, 0.5, 0.1, 0.2, -0.2, 0.3, 0.3, 0.2, -0.8, -0.9, 0.6, 0.4}
			case 3:
				return []float32{0.4, 0.6, 0.6, -0.4, 0.2}
			}
		}
		panic("NYI")
	}
}

type testNN interface {
	Graph() *ExprGraph
	X() *Node
	Pred() *Node
	Model() Nodes
}

type testTrainer struct {
	nn   testNN
	y    *Node
	cost *Node

	rnd *rand.Rand

	Solver Solver
	VM     VM
}

func newTestTrainer(nn testNN) (*testTrainer, error) {
	x := nn.X()
	ŷ := nn.Pred()
	g := nn.Graph()

	y := NewMatrix(g, x.Dtype(), WithShape(ŷ.Shape()[0], ŷ.Shape()[1]), WithName("Y"), WithInit(Zeroes()))
	cost := Must(Sum(Must(Square(Must(Sub(y, ŷ))))))

	_, err := Grad(cost, nn.Model()...)
	if err != nil {
		return nil, err
	}
	return &testTrainer{
		rnd:  rand.New(rand.NewSource(time.Now().UnixNano())),
		nn:   nn,
		y:    y,
		cost: cost,
	}, nil
}

func (t *testTrainer) Populate() error {
	var k *tensor.Dense

	// randomize x
	x := t.nn.X().Value().(*tensor.Dense)
	y := t.y.Value().(*tensor.Dense)
	dt := x.Dtype()
	switch dt {
	case tensor.Float64:
		data := x.Data().([]float64)
		for i := range data {
			data[i] = t.rnd.NormFloat64()
		}

		//	k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking(Gaussian64(0, 1, x.Shape()[1], y.Shape()[1])))
		k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking([]float64{-0.9, 1.2, 0.8, 1.1, 0.6}))
	case tensor.Float32:
		data := x.Data().([]float32)
		for i := range data {
			data[i] = float32(t.rnd.NormFloat64())
		}
		//k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking(Gaussian32(0, 1, x.Shape()[1], y.Shape()[1])))
		k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking([]float32{-0.9, 1.2, 0.8, 1.1, 0.6}))
	}

	log.Printf("%1.1f", x.Data())

	// compute x×k
	xk, err := x.MatMul(k)
	if err != nil {
		return err
	}

	// check shape
	if !y.Shape().Eq(xk.Shape()) {
		return errors.Errorf("Expected xk (%v) to have the same shape as y (%v).", xk.Shape(), y.Shape())
	}

	return tensor.Copy(y, xk)
}

func (t *testTrainer) populateDet() error {
	var k *tensor.Dense

	// randomize x
	x := t.nn.X().Value().(*tensor.Dense)
	y := t.y.Value().(*tensor.Dense)
	dt := x.Dtype()
	switch dt {
	case tensor.Float64:
		data := x.Data().([]float64)
		detX := []float64{-0.4, 1.1, 1.0, -0.3, 2.1, -1.9, -0.0, -0.9, -0.0, 0.7, -1.3, 1.2, -0.7, -0.1, -1.1, 0.8, -0.4, 1.0, 2.3, -1.0, -2.0, -0.7, -0.8, 0.2, 0.6, 1.5, -0.4, -0.1, -0.3, 0.1, -1.0, 0.4, -0.6, 0.8, -0.1, 0.8, -0.2, 1.6, -0.4, 0.1, 1.4, 0.7, -0.4, 2.1, -0.2, 0.6, 0.5, -1.0, 1.2, 0.6, 0.0, -0.8, -0.3, -0.7, 1.2, -0.2, -1.3, 1.1, -0.1, -1.4, -1.5, 1.8, -1.9, 1.5, -0.7, 0.6, 0.4, -0.2, -1.1, 0.9, 0.8, 0.8, -0.2, 1.1, -0.7, -1.4, 2.6, 0.2, 1.5, 0.3, 1.0, -0.2, 0.7, -0.9, -0.2, -1.3, 0.4, -0.3, 0.9, -0.3, -1.0, 0.7, 1.5, -0.9, 0.6, -0.8, -1.1, -1.4, 0.6, 0.2, 0.9, -0.0, 0.6, 0.5, 0.7, 1.5, 0.0, -1.0, -0.7, -0.3, -0.2, 1.4, 1.4, 0.4, -0.0, -1.6, 1.7, -0.4, -0.2, 0.2, -0.2, 1.1, -0.6, -0.5, -0.1, 0.6, 1.2, -0.1, 1.1, 0.2, 1.4, -1.3, -1.5, -0.7, -0.1, 0.5, 1.2, -0.8, -0.1, -0.0, 0.0, -1.3, -1.3, -1.4, -0.9, -1.0, 0.6, -1.5, 1.5, -2.8, 0.0, 0.9, 0.1, 1.4, -2.6, -1.2, 2.8, -0.5, -0.2, -0.5, 1.3, -0.6, -0.6, 0.1, -1.3, -0.5, 0.6, 0.4, 0.8, -0.9, 1.0, 0.3, -0.3, -0.4, 1.8, -0.0, -0.7, -0.5, 0.5, -0.1, 0.9, -1.1, -0.0, -0.2, 1.2, 1.2, -1.5, -1.0, -0.8, -0.2, -0.1, 0.4, -0.2, 0.4, -1.0, -0.3, -1.4, 1.4, 0.8, -0.1, -1.3, 0.7, -0.4, -1.3, -0.2, 1.1, 2.0, 1.0, -0.8, -0.8, 0.4, -0.3, 0.8, -0.4, 0.4, -2.9, 1.1, 0.5, -1.9, 0.5, -0.3, -0.2, -0.7, -0.7, -0.4, 0.7, -0.3, 1.5, 0.7, -0.5, 0.6, -0.9, -0.3, -1.1, 0.6, 0.3, -1.4, -2.3, 0.0, -1.2, 1.0, -0.5, -0.4, 1.7, -0.8, -0.2, 0.1, 0.0, -0.5, -0.2, 0.3, 1.0, 0.2, 2.7, 0.2, 0.3, 1.8, -0.6, -1.6, 0.4, -0.7, -1.4, -1.8, -1.1, -2.3, -0.0, -1.4, 0.7, -1.3, 2.6, -1.4, -0.6, 0.7, -0.4, -2.0, 1.5, 0.3, 0.7, 1.9, 1.9, -0.5, -0.5, -0.2, 0.2, -0.1, 1.0, 0.4, -1.3, 2.3, 1.4, 0.5, 1.7, -1.2, 0.9, 0.3, 0.5, -0.6, -1.4, 1.6, 0.5, -0.1, 0.5, -1.8, -0.1, -0.2, 0.5, -1.4, -1.3, -0.0, 0.6, -0.4, -0.5, -2.4, 1.8, 1.1, -2.1, -0.7, 1.5, -0.9, 1.6}
		copy(data, detX)

		k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking([]float64{-0.9, 1.2, 0.8, 1.1, 0.6}))
	case tensor.Float32:
		data := x.Data().([]float32)
		detX := []float32{-0.4, 1.1, 1.0, -0.3, 2.1, -1.9, -0.0, -0.9, -0.0, 0.7, -1.3, 1.2, -0.7, -0.1, -1.1, 0.8, -0.4, 1.0, 2.3, -1.0, -2.0, -0.7, -0.8, 0.2, 0.6, 1.5, -0.4, -0.1, -0.3, 0.1, -1.0, 0.4, -0.6, 0.8, -0.1, 0.8, -0.2, 1.6, -0.4, 0.1, 1.4, 0.7, -0.4, 2.1, -0.2, 0.6, 0.5, -1.0, 1.2, 0.6, 0.0, -0.8, -0.3, -0.7, 1.2, -0.2, -1.3, 1.1, -0.1, -1.4, -1.5, 1.8, -1.9, 1.5, -0.7, 0.6, 0.4, -0.2, -1.1, 0.9, 0.8, 0.8, -0.2, 1.1, -0.7, -1.4, 2.6, 0.2, 1.5, 0.3, 1.0, -0.2, 0.7, -0.9, -0.2, -1.3, 0.4, -0.3, 0.9, -0.3, -1.0, 0.7, 1.5, -0.9, 0.6, -0.8, -1.1, -1.4, 0.6, 0.2, 0.9, -0.0, 0.6, 0.5, 0.7, 1.5, 0.0, -1.0, -0.7, -0.3, -0.2, 1.4, 1.4, 0.4, -0.0, -1.6, 1.7, -0.4, -0.2, 0.2, -0.2, 1.1, -0.6, -0.5, -0.1, 0.6, 1.2, -0.1, 1.1, 0.2, 1.4, -1.3, -1.5, -0.7, -0.1, 0.5, 1.2, -0.8, -0.1, -0.0, 0.0, -1.3, -1.3, -1.4, -0.9, -1.0, 0.6, -1.5, 1.5, -2.8, 0.0, 0.9, 0.1, 1.4, -2.6, -1.2, 2.8, -0.5, -0.2, -0.5, 1.3, -0.6, -0.6, 0.1, -1.3, -0.5, 0.6, 0.4, 0.8, -0.9, 1.0, 0.3, -0.3, -0.4, 1.8, -0.0, -0.7, -0.5, 0.5, -0.1, 0.9, -1.1, -0.0, -0.2, 1.2, 1.2, -1.5, -1.0, -0.8, -0.2, -0.1, 0.4, -0.2, 0.4, -1.0, -0.3, -1.4, 1.4, 0.8, -0.1, -1.3, 0.7, -0.4, -1.3, -0.2, 1.1, 2.0, 1.0, -0.8, -0.8, 0.4, -0.3, 0.8, -0.4, 0.4, -2.9, 1.1, 0.5, -1.9, 0.5, -0.3, -0.2, -0.7, -0.7, -0.4, 0.7, -0.3, 1.5, 0.7, -0.5, 0.6, -0.9, -0.3, -1.1, 0.6, 0.3, -1.4, -2.3, 0.0, -1.2, 1.0, -0.5, -0.4, 1.7, -0.8, -0.2, 0.1, 0.0, -0.5, -0.2, 0.3, 1.0, 0.2, 2.7, 0.2, 0.3, 1.8, -0.6, -1.6, 0.4, -0.7, -1.4, -1.8, -1.1, -2.3, -0.0, -1.4, 0.7, -1.3, 2.6, -1.4, -0.6, 0.7, -0.4, -2.0, 1.5, 0.3, 0.7, 1.9, 1.9, -0.5, -0.5, -0.2, 0.2, -0.1, 1.0, 0.4, -1.3, 2.3, 1.4, 0.5, 1.7, -1.2, 0.9, 0.3, 0.5, -0.6, -1.4, 1.6, 0.5, -0.1, 0.5, -1.8, -0.1, -0.2, 0.5, -1.4, -1.3, -0.0, 0.6, -0.4, -0.5, -2.4, 1.8, 1.1, -2.1, -0.7, 1.5, -0.9, 1.6}
		copy(data, detX)
		k = tensor.New(tensor.WithShape(x.Shape()[1], y.Shape()[1]), tensor.WithBacking([]float32{-0.9, 1.2, 0.8, 1.1, 0.6}))
	}

	// compute x×k
	xk, err := x.MatMul(k)
	if err != nil {
		return err
	}

	// check shape
	if !y.Shape().Eq(xk.Shape()) {
		return errors.Errorf("Expected xk (%v) to have the same shape as y (%v).", xk.Shape(), y.Shape())
	}

	return tensor.Copy(y, xk)
}

func toFloat64(a interface{}) float64 {
	switch at := a.(type) {
	case float32:
		return float64(at)
	case float64:
		return at
	}
	panic("Unreachable")
}

func adam2(t *testing.T, iterN int) int {
	nn := randTestNN3(true)
	train, err := newTestTrainer(nn)
	if err != nil {
		t.Fatal(err)
	}
	train.Solver = NewAdamSolver()
	train.VM = NewTapeMachine(train.nn.Graph(), BindDualValues(), TraceExec())

	if err = train.populateDet(); err != nil {
		t.Fatal(err)
	}
	model := nn.Model()
	m := make([]ValueGrad, 0, len(model))
	for _, n := range model {
		m = append(m, n)
	}

	var errs []float64
	for i := 0; i < 3; i++ {
		if err = train.VM.RunAll(); err != nil {
			t.Fatalf("Error while running VM")
		}
		if err = train.Solver.Step(m); err != nil {
			t.Fatalf("Error while stepping %v", err)
		}
		train.VM.Reset()

		errs = append(errs, toFloat64(train.cost.Value().Data()))
	}

	if iterN == 1 {
		t.Logf("%v", errs)
	}

	// find the iter in which error is < 1e-5
	var iter int = -1
	for i := range errs {
		if errs[i] < 1e-5 {
			iter = i
			break
		}
	}
	return iter
}

func TestAdam2(t *testing.T) {
	var iters []int
	for i := 0; i < 1; i++ {
		iter := adam2(t, i)
		iters = append(iters, iter)
		runtime.GC()
	}
	t.Logf("%v", iters)
}
