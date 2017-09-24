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
	m := NewTapeMachine(g, TraceExec(), BindDualValues())
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

var im2colTests = []struct {
	kernel tensor.Shape
	pad    tensor.Shape
	stride tensor.Shape
}{
	{tensor.Shape{4, 4}, tensor.Shape{0, 0}, tensor.Shape{1, 1}},
	{tensor.Shape{3, 3}, tensor.Shape{1, 1}, tensor.Shape{2, 2}},
	{tensor.Shape{3, 3}, tensor.Shape{1, 1}, tensor.Shape{3, 3}},
}

func im2colTest(t *testing.T, dt tensor.Dtype, kernel, pad, stride tensor.Shape) {
	g := NewGraph()
	x := NewTensor(g, dt, 4, WithShape(2, 1, 28, 28), WithInit(RangedFrom(0))) // mnist, in batches of 10
	y, err := Im2Col(x, kernel, pad, stride)
	if err != nil {
		t.Error(err)
		return
	}
	cost := Must(Sum(y))

	grads, err := Grad(cost, x)
	if err != nil {
		t.Errorf("error while Grad(): %v", err)
		return
	}

	m := NewTapeMachine(g, BindDualValues())
	if err := m.RunAll(); err != nil {
		t.Error(err)
		return
	}
	t.Logf("x: %v", x.Value())
	t.Logf("c: %3.3f", cost.Value())
	t.Logf("xG: %v", grads[0])
}

func TestIm2Col(t *testing.T) {
	// assert := assert.New(t)
	dts := []tensor.Dtype{tensor.Float64, tensor.Float32}
	for _, dt := range dts {
		for _, i2ct := range im2colTests {
			im2colTest(t, dt, i2ct.kernel, i2ct.pad, i2ct.stride)
		}
	}
}
