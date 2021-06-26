package gorgonia

import (
	"fmt"
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestDropoutAll(t *testing.T) {
	var tests = []struct {
		dt           tensor.Dtype
		prob         float64
		rand         []float64
		expected     interface{}
		expectedGrad interface{}
	}{
		{Float64, 0.0, []float64{0.0, 0.2, 0.5, 0.8, 1.0}, []float64{1.0, 1.0, 1.0, 1.0, 1.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}},
		{Float64, 0.2, []float64{0.0, 0.2, 0.5, 0.8, 1.0}, []float64{1.25, 1.25, 1.25, 0.0, 0.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}},
		{Float64, 0.5, []float64{0.0, 0.2, 0.5, 0.8, 1.0}, []float64{2.0, 2.0, 0.0, 0.0, 0.0}, []float64{0.2, 0.2, 0.2, 0.2, 0.2}},
		{Float32, 0.2, []float64{0.0, 0.2, 0.5, 0.8, 1.0}, []float32{1.25, 1.25, 1.25, 0.0, 0.0}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}},
		{Float32, 0.5, []float64{0.0, 0.2, 0.5, 0.8, 1.0}, []float32{2.0, 2.0, 0.0, 0.0, 0.0}, []float32{0.2, 0.2, 0.2, 0.2, 0.2}},
	}

	for _, tt := range tests {
		name := fmt.Sprintf("%v-%.1f", tt.dt, tt.prob)
		t.Run(name, func(t *testing.T) {
			randCount := 0
			randFn := func() float64 {
				v := tt.rand[randCount%len(tt.rand)]
				randCount++

				return v
			}

			g := NewGraph()
			x := NewVector(g, tt.dt, WithShape(5), WithName("x"), WithInit(Ones()))

			y, err := ApplyOp(newDropoutOp(tt.prob, randFn), x)
			assert.NoError(t, err)

			cost, _ := Mean(y)
			if _, err := Grad(cost, x); err != nil {
				t.Fatal(err)
			}

			m := NewTapeMachine(g, BindDualValues())
			defer m.Close()
			defer runtime.GC()

			require.NoError(t, m.RunAll())
			assert.Equal(t, tt.expected, y.Value().Data())

			yGrad, err := y.Grad()
			require.NoError(t, err)

			assert.Equal(t, tt.expectedGrad, yGrad.Data())
		})
	}
}

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

func TestDropout_integration(t *testing.T) {
	if err := dropoutTest(t, Float64); err != nil {
		t.Errorf("%+v", err)
	}

	if err := dropoutTest(t, Float32); err != nil {
		t.Errorf("%+v", err)
	}

	// visual inspection
	// ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
}
