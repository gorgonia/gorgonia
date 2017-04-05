package gorgonia

import (
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func dropoutTest(t *testing.T, dt tensor.Dtype) error {
	g := NewGraph()
	x := NewVector(g, dt, WithShape(10), WithName("x"), WithInit(RangedFrom(0)))
	w := NewMatrix(g, dt, WithShape(20, 10), WithName("w"), WithInit(RangedFrom(0)))
	w2 := NewMatrix(g, dt, WithShape(10, 20), WithName("w2"), WithInit(RangedFrom(0)))
	wx := Must(Mul(w, x))
	act := Must(Cube(wx))
	do := Must(Dropout(act, 0.5))

	act2 := Must(Cube(Must(Mul(w2, do))))
	do2 := Must(Dropout(act2, 0.1))
	cost := Must(Sum(do2))

	_, err := Grad(cost, x, w, w2)

	if err != nil {
		ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
		// t.Fatalf("%+v", err)
		return err
	}

	// logger := log.New(os.Stderr, "", 0)

	// m := NewTapeMachine(g, TraceExec(), BindDualValues(), WithLogger(logger), WithWatchlist())
	m := NewTapeMachine(g, TraceExec(), BindDualValues(), UseCudaFor())
	cudaLogf("%v", m.Prog())
	defer runtime.GC()
	if err := m.RunAll(); err != nil {
		return err
	}
	return nil
}

func TestDropout(t *testing.T) {
	// t.Skip()

	if err := dropoutTest(t, Float64); err != nil {
		t.Errorf("%+v", err)
	}

	if err := dropoutTest(t, Float32); err != nil {
		t.Errorf("%+v", err)
	}

	// visual inspection
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
}
