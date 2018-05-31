package gorgonia

import (
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/dawson"
	"gorgonia.org/tensor"
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
	kernel   tensor.Shape
	pad      tensor.Shape
	stride   tensor.Shape
	dilation tensor.Shape
}{
	{tensor.Shape{4, 4}, tensor.Shape{0, 0}, tensor.Shape{1, 1}, tensor.Shape{1, 1}},
	{tensor.Shape{3, 3}, tensor.Shape{1, 1}, tensor.Shape{2, 2}, tensor.Shape{1, 1}},
	{tensor.Shape{3, 3}, tensor.Shape{1, 1}, tensor.Shape{3, 3}, tensor.Shape{1, 1}},
}

func im2colTest(t *testing.T, dt tensor.Dtype, kernel, pad, stride, dilation tensor.Shape) {
	assert := assert.New(t)
	g := NewGraph()
	x := NewTensor(g, dt, 4, WithShape(2, 1, 28, 28), WithInit(RangedFrom(0))) // mnist, in batches of 10
	y, err := Im2Col(x, kernel, pad, stride, dilation)
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
	// t.Logf("x: %v", x.Value())
	// t.Logf("c: %3.3f", cost.Value())
	// t.Logf("xG: %v", grads[0].Value())

	h := NewGraph()
	a := NewTensor(h, dt, 4, WithShape(2, 1, 28, 28), WithInit(RangedFrom(0)))
	b, err := Im2Col(a, kernel, pad, stride, dilation)
	if err != nil {
		t.Error(err)
		return
	}
	cost2 := Must(Sum(b))
	n := NewLispMachine(h)
	if err = n.RunAll(); err != nil {
		t.Error(err)
		return
	}
	aG, err := a.Grad()
	if err != nil {
		t.Error(err)
		return
	}

	// t.Logf("a: %v", a.Value())
	// t.Logf("c: %3.3f", cost2.Value())
	// t.Logf("aG: %v", aG)

	assert.Equal(x.Value().Data(), a.Value().Data())
	assert.Equal(grads[0].Value().Data(), aG.Data())
	assert.Equal(cost.Value().Data(), cost2.Value().Data())
}

func TestIm2Col(t *testing.T) {
	// assert := assert.New(t)
	dts := []tensor.Dtype{tensor.Float64, tensor.Float32}
	for _, dt := range dts {
		for _, i2ct := range im2colTests {
			im2colTest(t, dt, i2ct.kernel, i2ct.pad, i2ct.stride, i2ct.dilation)
		}
	}
}

func TestMaxPool2D(t *testing.T) {
	assert := assert.New(t)
	dts := []tensor.Dtype{tensor.Float64, tensor.Float32}
	for _, dt := range dts {
		g := NewGraph()
		x := NewTensor(g, dt, 4, WithShape(1, 2, 3, 4), WithInit(RangedFrom(0)))
		y, err := MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		cost := Must(Sum(y))
		grads, err := Grad(cost, x)
		if err != nil {
			t.Fatal(err)
		}

		m := NewTapeMachine(g, BindDualValues())
		if err := m.RunAll(); err != nil {
			t.Fatal(err)
		}

		// t.Logf("x %v", x.Value())
		// t.Logf("y: %v", y.Value())
		// t.Logf("c: %v", cost.Value())
		// t.Logf("xG: %v", grads[0])

		h := NewGraph()
		a := NewTensor(h, dt, 4, WithShape(1, 2, 3, 4), WithInit(RangedFrom(0)))
		b, err := MaxPool2D(a, tensor.Shape{2, 2}, []int{0, 0}, []int{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		cost2 := Must(Sum(b))
		if err != nil {
			t.Fatal(err)
		}

		m2 := NewLispMachine(h)
		if err = m2.RunAll(); err != nil {
			t.Fatal(err)
		}
		aG, err := a.Grad()
		if err != nil {
			t.Error(err)
			return
		}

		assert.Equal(x.Value().Data(), a.Value().Data())
		assert.Equal(grads[0].Value().Data(), aG.Data())
		assert.Equal(cost.Value().Data(), cost2.Value().Data())

	}

}

/*
func TestDumb(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, Float32, 4, WithShape(10, 128, 6, 6), WithInit(RangedFrom(0)))
	// x := NewTensor(g, Float32, 4, WithShape(10, 128, 6, 6), WithName("x"))
	p := Must(MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2}))
	r := Must(Reshape(p, tensor.Shape{10, 512}))
	c := Must(Sum(r))
	Grad(c, x)
	// ioutil.WriteFile("dumbdumb.dot", []byte(g.ToDot()), 0644)
	// prog, _, _ := Compile(g)
	// log.Printf("%v", prog)
	logger := log.New(os.Stderr, "", 0)
	m := NewTapeMachine(g, WithLogger(logger), WithWatchlist(), WithValueFmt("%+s"))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
}
*/

func TestBatchNorm_F64(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, Float64, 4, WithShape(5, 2, 3, 4), WithInit(Gaussian(0, 1)))
	y, op, err := BatchNorm(x, 0.9, 1e-5, true)
	if err != nil {
		t.Fatal(err)
	}

	var yVal Value
	Read(y, &yVal)

	cost, _ := Mean(y)

	if _, err := Grad(cost, x); err != nil {
		ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m := NewTapeMachine(g, BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}

	shape := x.Shape()
	n, c, h, w := shape[0], shape[1], shape[2], shape[3]

	yVT := yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float64
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float64)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float64(h * w * n)
		variance /= float64(h * w * n)

		if !dawson.ToleranceF64(sum, 0, 0.001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 1, 0.001) {
			t.Errorf("channel %d: Expected variance to be near 1. Got %v", j, variance)
		}
	}

	op.SetTesting()
	m = NewTapeMachine(g, BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	yVT = yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float64
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float64)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float64(h * w * n)
		variance /= float64(h * w * n)

		if !dawson.ToleranceF64(sum, 0, 0.001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 0.9833, 0.001) {
			t.Errorf("channel %d: Expected variance to be near 0.98. Got %v", j, variance)
		}
	}

}

func TestBatchNorm_F32(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, Float32, 4, WithShape(5, 2, 3, 4), WithInit(Gaussian(0, 1)))
	y, op, err := BatchNorm(x, 0.9, 1e-5, true)
	if err != nil {
		t.Fatal(err)
	}

	var yVal Value
	Read(y, &yVal)

	cost, _ := Mean(y)

	if _, err := Grad(cost, x); err != nil {
		ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m := NewTapeMachine(g, BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}

	shape := x.Shape()
	n, c, h, w := shape[0], shape[1], shape[2], shape[3]

	yVT := yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float32
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float32)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float32(h * w * n)
		variance /= float32(h * w * n)

		if !dawson.ToleranceF32(sum, 0, 0.001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF32(variance, 1, 0.001) {
			t.Errorf("channel %d: Expected variance to be near 1. Got %v", j, variance)
		}
	}

	op.SetTesting()
	m = NewTapeMachine(g, BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	yVT = yVal.(*tensor.Dense)
	for j := 0; j < c; j++ {
		var sum, variance float32
		for i := 0; i < n; i++ {
			for k := 0; k < h; k++ {
				for l := 0; l < w; l++ {
					at, err := yVT.At(i, j, k, l)
					if err != nil {
						t.Fatal(err)
					}
					atf := at.(float32)
					sum += atf
					variance += atf * atf
				}
			}
		}
		sum /= float32(h * w * n)
		variance /= float32(h * w * n)

		if !dawson.ToleranceF32(sum, 0, 0.001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF32(variance, 0.9833, 0.001) {
			t.Errorf("channel %d: Expected variance to be near 0.98. Got %v", j, variance)
		}
	}

}
