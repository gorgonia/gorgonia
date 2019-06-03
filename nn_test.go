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
	defer m.Close()
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
	defer m.Close()
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
	defer n.Close()
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

		m.Close()
		m2.Close()
	}

}

func TestBatchNorm_F64(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, Float64, 4, WithShape(5, 2, 3, 4), WithInit(Gaussian(0, 1)), WithName("x"))
	scale := NewTensor(g, Float64, 4, WithShape(5, 2, 3, 4), WithInit(Ones()), WithName("scale"))
	bias := NewTensor(g, Float64, 4, WithShape(5, 2, 3, 4), WithInit(Zeroes()), WithName("bias"))
	y, _, _, op, err := BatchNorm(x, scale, bias, 0.9, 1e-5)
	if err != nil {
		t.Fatal(err)
	}

	var yVal Value
	Read(y, &yVal)

	cost, _ := Mean(y)

	if _, err := Grad(cost, x); err != nil {
		t.Fatal(err)
	}

	m := NewTapeMachine(g, BindDualValues(x), TraceExec())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	m.Close()
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)

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

		if !dawson.ToleranceF64(sum, 0, 0.00001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 1, 0.0001) {
			t.Errorf("channel %d: Expected variance to be near 1. Got %v", j, variance)
		}
	}

	op.SetTesting()
	m = NewTapeMachine(g, BindDualValues(x))
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	m.Close()
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

		if !dawson.ToleranceF64(sum, 0, 0.00001) {
			t.Errorf("channel %d: Expected sum to be near 0. Got %v", j, sum)
		}

		if !dawson.ToleranceF64(variance, 0.9833, 0.0001) {
			t.Errorf("channel %d: Expected variance to be near 0.98. Got %v", j, variance)
		}
	}

}

func TestBatchNorm_F32(t *testing.T) {
	g := NewGraph()
	x := NewTensor(g, Float32, 4, WithShape(5, 2, 3, 4), WithInit(Gaussian(0, 1)))
	scale := NewTensor(g, Float32, 4, WithShape(5, 2, 3, 4), WithInit(Ones()), WithName("scale"))
	bias := NewTensor(g, Float32, 4, WithShape(5, 2, 3, 4), WithInit(Zeroes()), WithName("bias"))
	y, _, _, op, err := BatchNorm(x, scale, bias, 0.9, 1e-5)
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
	m.Close()

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
	m.Close()
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

func TestLeakyRelu(t *testing.T) {
	tests := []struct {
		name  string
		alpha float64
		xT    tensor.Tensor
		yT    tensor.Tensor
	}{
		{
			name:  "simple float32",
			alpha: 0.1,
			xT: tensor.New(
				tensor.WithShape(3, 4, 5),
				tensor.WithBacking([]float32{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.9772779, 0.95008844, -0.1513572, -0.10321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.20515826, 0.3130677, -0.85409576, -2.5529897, 0.6536186, 0.8644362, -0.742165, 2.2697546, -1.4543657, 0.045758516, -0.18718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.88778573, -1.9807965, -0.34791216, 0.15634897, 1.2302907, 1.2023798, -0.3873268, -0.30230275, -1.048553, -1.420018, -1.7062702, 1.9507754, -0.5096522, -0.4380743, -1.2527953, 0.7774904, -1.6138978, -0.21274029, -0.89546657, 0.3869025, -0.51080513, -1.1806322, -0.028182229, 0.42833188, 0.06651722, 0.3024719, -0.6343221, -0.36274117}),
			),
			yT: tensor.New(
				tensor.WithShape(3, 4, 5),
				tensor.WithBacking([]float32{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.09772779, 0.95008844, -0.01513572, -0.010321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.020515827, 0.3130677, -0.085409574, -0.25529897, 0.6536186, 0.8644362, -0.07421651, 2.2697546, -0.14543657, 0.045758516, -0.018718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.08877858, -0.19807965, -0.034791216, 0.15634897, 1.2302907, 1.2023798, -0.03873268, -0.030230274, -0.1048553, -0.1420018, -0.17062703, 1.9507754, -0.05096522, -0.04380743, -0.12527953, 0.7774904, -0.16138978, -0.021274028, -0.08954666, 0.3869025, -0.051080514, -0.11806323, -0.0028182229, 0.42833188, 0.06651722, 0.3024719, -0.06343221, -0.036274116}),
			),
		},
		{
			name:  "simple float64",
			alpha: 0.1,
			xT: tensor.New(
				tensor.WithShape(3, 4, 5),
				tensor.WithBacking([]float64{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.9772779, 0.95008844, -0.1513572, -0.10321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.20515826, 0.3130677, -0.85409576, -2.5529897, 0.6536186, 0.8644362, -0.742165, 2.2697546, -1.4543657, 0.045758516, -0.18718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.88778573, -1.9807965, -0.34791216, 0.15634897, 1.2302907, 1.2023798, -0.3873268, -0.30230275, -1.048553, -1.420018, -1.7062702, 1.9507754, -0.5096522, -0.4380743, -1.2527953, 0.7774904, -1.6138978, -0.21274029, -0.89546657, 0.3869025, -0.51080513, -1.1806322, -0.028182229, 0.42833188, 0.06651722, 0.3024719, -0.6343221, -0.36274117}),
			),
			yT: tensor.New(
				tensor.WithShape(3, 4, 5),
				tensor.WithBacking([]float64{1.7640524, 0.4001572, 0.978738, 2.2408931, 1.867558, -0.09772779, 0.95008844, -0.01513572, -0.010321885, 0.41059852, 0.14404356, 1.4542735, 0.7610377, 0.121675014, 0.44386324, 0.33367434, 1.4940791, -0.020515827, 0.3130677, -0.085409574, -0.25529897, 0.6536186, 0.8644362, -0.07421651, 2.2697546, -0.14543657, 0.045758516, -0.018718386, 1.5327792, 1.4693588, 0.15494743, 0.37816253, -0.08877858, -0.19807965, -0.034791216, 0.15634897, 1.2302907, 1.2023798, -0.03873268, -0.030230274, -0.1048553, -0.1420018, -0.17062703, 1.9507754, -0.05096522, -0.04380743, -0.12527953, 0.7774904, -0.16138978, -0.021274028, -0.08954666, 0.3869025, -0.051080514, -0.11806323, -0.0028182229, 0.42833188, 0.06651722, 0.3024719, -0.06343221, -0.036274116}),
			),
		},
	}
	for _, tst := range tests {
		t.Run(tst.name, func(t *testing.T) {
			g := NewGraph()
			assert := assert.New(t)
			x := NodeFromAny(g, tst.xT)
			output, err := LeakyRelu(x, tst.alpha)

			if err != nil {
				t.Fatalf("%+v", err)
			}
			m := NewTapeMachine(g)
			if err := m.RunAll(); err != nil {
				t.Fatalf("%+v", err)
			}
			defer m.Close()
			assert.InDeltaSlice(tst.yT.Data(), output.Value().Data(), 1e-6, "the two tensors should be equal.")
		})
	}
}
