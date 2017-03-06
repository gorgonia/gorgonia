// +build cuda

package gorgonia

import (
	"io/ioutil"
	"runtime"
	"testing"

	"github.com/alecthomas/assert"
	"github.com/chewxy/gorgonia/tensor"
)

func TestCUDACube(t *testing.T) {
	defer runtime.GC()

	assert := assert.New(t)
	xT := tensor.New(tensor.Of(tensor.Float32), tensor.WithBacking(tensor.Range(Float32, 0, 32)), tensor.WithShape(8, 4))

	g := NewGraph(WithGraphName("Test"))
	x := NewMatrix(g, tensor.Float32, WithName("x"), WithShape(8, 4), WithValue(xT))
	x3 := Must(Cube(x))

	prog, locMap, err := Compile(g)
	// t.Logf("Prog: \n%v", prog)
	if err != nil {
		t.Fatal(err)
	}
	m := NewTapeMachine(prog, locMap, UseCudaFor())
	if err = m.LoadCUDAFunc("cube32", cube32PTX); err != nil {
		t.Fatal(err)
	}
	if err = m.RunAll(); err != nil {
		t.Error(err)
	}
	correct := []float32{0, 1, 8, 27, 64, 125, 216, 343, 512, 729, 1000, 1331, 1728, 2197, 2744, 3375, 4096, 4913, 5832, 6859, 8000, 9261, 10648, 12167, 13824, 15625, 17576, 19683, 21952, 24389, 27000, 29791}
	assert.Equal(correct, x3.Value().Data())

	correct = tensor.Range(tensor.Float32, 0, 32).([]float32)
	assert.Equal(correct, x.Value().Data())
}

func TestCUDABasicArithmetic(t *testing.T) {
	assert := assert.New(t)
	for i, bot := range binOpTests {
		g := NewGraph()
		xV, _ := CloneValue(bot.a)
		yV, _ := CloneValue(bot.b)
		x := NodeFromAny(g, xV, WithName("x"))
		y := NodeFromAny(g, yV, WithName("y"))

		var ret *Node
		var err error
		if ret, err = bot.binOp(x, y); err != nil {
			t.Errorf("Test %d: %v", i, err)
			runtime.GC()
			continue
		}

		cost := Must(Sum(ret))
		var grads Nodes
		if grads, err = Grad(cost, x, y); err != nil {
			t.Errorf("Test %d: error while symbolic op: %v", i, err)
			runtime.GC()
			continue
		}

		prog, locMap, err := Compile(g)
		// t.Log(prog)
		// t.Log(locMap)
		if err != nil {
			t.Errorf("Test %d: error while compiling: %v", i, err)
			runtime.GC()
			continue
		}

		// logger := log.New(os.Stderr, "", 0)
		// m1 := NewTapeMachine(prog, locMap, TraceExec(), UseCudaFor(), WithLogger(logger), WithWatchlist())
		m1 := NewTapeMachine(prog, locMap, TraceExec(), UseCudaFor())
		if err = m1.RunAll(); err != nil {
			t.Errorf("Test %d: error while running %v", i, err)
			runtime.GC()
			continue
		}

		ioutil.WriteFile("add.dot", []byte(g.ToDot()), 0644)

		assert.Equal(bot.correct.Data(), ret.Value().Data(), "i %d | %v | %v", i, bot.correct.Data(), ret.Value())
		assert.True(bot.correctShape.Eq(ret.Shape()))
		assert.Equal(2, len(grads))
		assert.Equal(bot.correctDerivA.Data(), grads[0].Value().Data(), "Test %v", i)
		assert.Equal(bot.correctDerivB.Data(), grads[1].Value().Data())
		runtime.GC()
	}
}
