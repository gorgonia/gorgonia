// +build cuda

package gorgonia

import (
	"log"
	"os"
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestCUDACube(t *testing.T) {
	defer runtime.GC()

	assert := assert.New(t)
	xT := tensor.New(tensor.Of(tensor.Float32), tensor.WithBacking(tensor.Range(Float32, 0, 32)), tensor.WithShape(8, 4))

	g := NewGraph(WithGraphName("Test"))
	x := NewMatrix(g, tensor.Float32, WithName("x"), WithShape(8, 4), WithValue(xT))
	x3 := Must(Cube(x))
	var x3Val Value
	Read(x3, &x3Val)

	m := NewTapeMachine(g)
	defer runtime.GC()
	if err := m.RunAll(); err != nil {
		t.Error(err)
	}
	correct := []float32{0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952, 24389, 27000, 29791}
	assert.Equal(correct, x3Val.Data())

	t.Logf("0x%x", x3Val.Uintptr())
	t.Logf("\n%v", m.cpumem[1])
	t.Logf("0x%x", m.cpumem[1].Uintptr())

	correct = tensor.Range(tensor.Float32, 0, 32).([]float32)
	assert.Equal(correct, x.Value().Data())
}

func TestCUDABasicArithmetic(t *testing.T) {
	assert := assert.New(t)
	for i, bot := range binOpTests {
		// log.Printf("TEST %d", i)
		// if i != 2 {
		// 	continue
		// }

		g := NewGraph()
		xV, _ := CloneValue(bot.a)
		yV, _ := CloneValue(bot.b)
		x := NodeFromAny(g, xV, WithName("x"))
		y := NodeFromAny(g, yV, WithName("y"))

		var ret *Node
		var retVal Value
		var err error
		if ret, err = bot.binOp(x, y); err != nil {
			t.Errorf("Test %d: %v", i, err)
			runtime.GC()
			continue
		}
		Read(ret, &retVal)

		cost := Must(Sum(ret))
		var grads Nodes
		if grads, err = Grad(cost, x, y); err != nil {
			t.Errorf("Test %d: error while symbolic op: %v", i, err)
			runtime.GC()
			continue
		}

		// logger := log.New(os.Stderr, "", 0)
		// m1 := NewTapeMachine(g, UseCudaFor(), WithLogger(logger), WithWatchlist())
		m1 := NewTapeMachine(g)
		if err = m1.RunAll(); err != nil {
			t.Logf("%v", m1.Prog())
			t.Fatalf("Test %d: error while running %+v", i, err)
			runtime.GC()
			continue
		}

		as := newAssertState(assert)
		as.Equal(bot.correct.Data(), retVal.Data(), "Test %d result", i)
		as.True(bot.correctShape.Eq(ret.Shape()))
		as.Equal(2, len(grads))
		as.Equal(bot.correctDerivA.Data(), grads[0].Value().Data(), "Test %v xgrad", i)
		as.Equal(bot.correctDerivB.Data(), grads[1].Value().Data(), "Test %v ygrad. Expected %v. Got %v", i, bot.correctDerivB, grads[1].Value())
		if !as.cont {
			prog := m1.Prog()
			t.Fatalf("Test %d failed. Prog: %v", i, prog)
		}
		runtime.GC()
	}

	// _logger_ = spare
}

func TestMultiDeviceArithmetic(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithName("x"), WithShape(2, 2))
	y := NewMatrix(g, Float64, WithName("y"), WithShape(2, 2))
	z := Must(Add(x, y))
	zpx := Must(Add(x, z)) // z would be on device
	Must(Sum(zpx))

	xV := tensor.New(tensor.WithBacking([]float64{0, 1, 2, 3}), tensor.WithShape(2, 2))
	yV := tensor.New(tensor.WithBacking([]float64{0, 1, 2, 3}), tensor.WithShape(2, 2))

	Let(x, xV)
	Let(y, yV)

	logger := log.New(os.Stderr, "", 0)
	m := NewLispMachine(g, WithLogger(logger), WithWatchlist(), LogBothDir())

	t.Logf("zpx.Device: %v", zpx.Device())
	t.Logf("x.Device: %v", x.Device())
	t.Logf("y.Device: %v", y.Device())

	if err := m.RunAll(); err != nil {
		t.Errorf("err: %+v", err)
	}

}
