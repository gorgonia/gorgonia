package gorgonia

import (
	"bytes"
	"log"
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/stretchr/testify/assert"
)

func TestLispMachineBasics(t *testing.T) {
	assert := assert.New(t)
	var m *lispMachine
	// var err error
	var buf bytes.Buffer

	// test various flags first
	m = NewLispMachine(nil)
	assert.Equal(byte(0x3), m.runFlags)
	assert.True(m.runFwd())
	assert.True(m.runBwd())

	logger := log.New(&buf, "", 0)
	m = NewLispMachine(nil, WithLogger(logger))
	assert.Equal(logger, m.logger)
	assert.Equal(byte(0x0), m.logFlags) // if you pass in a logger without telling which direction to log... nothing gets logged

	m = NewLispMachine(nil, WithLogger(nil))
	assert.NotNil(m.logger)

	m = NewLispMachine(nil, WithValueFmt("%v"))
	assert.Equal("%v", m.valueFmt)

	m = NewLispMachine(nil, WithNaNWatch())
	assert.Equal(byte(0x7), m.runFlags)
	assert.True(m.watchNaN())

	m = NewLispMachine(nil, WithInfWatch())
	assert.Equal(byte(0xb), m.runFlags)
	assert.True(m.watchInf())

	m = NewLispMachine(nil, ExecuteFwdOnly())
	assert.Equal(byte(0x1), m.runFlags)
	assert.True(m.runFwd())
	assert.False(m.runBwd())

	m = NewLispMachine(nil, ExecuteBwdOnly())
	assert.Equal(byte(0x2), m.runFlags)
	assert.True(m.runBwd())
	assert.False(m.runFwd())

	m = NewLispMachine(nil, LogFwd())
	assert.Equal(byte(0x1), m.logFlags)
	assert.Equal(byte(0x3), m.runFlags)
	assert.True(m.logFwd())
	assert.False(m.logBwd())

	m = NewLispMachine(nil, LogBwd())
	assert.Equal(byte(0x2), m.logFlags)
	assert.Equal(byte(0x3), m.runFlags)
	assert.True(m.logBwd())
	assert.False(m.logFwd())

	// if you pass in a watchlist, but don't have any logger, well, it's not gonna log anything
	m = NewLispMachine(nil, WithWatchlist())
	assert.Equal(byte(0x80), m.logFlags)
	assert.Equal(byte(0x3), m.runFlags)
	assert.True(m.watchAll())

}

func TestLispMachineMechanics(t *testing.T) {
	assert := assert.New(t)
	var err error
	g, x, y, z := simpleVecEqn()

	sz := Must(Sum(z))

	xBack := []float64{1, 5}
	yBack := []float64{2, 4}
	Let(x, tf64.NewTensor(tf64.WithShape(x.shape...), tf64.WithBacking(xBack)))
	Let(y, tf64.NewTensor(tf64.WithShape(y.shape...), tf64.WithBacking(yBack)))

	machine := NewLispMachine(g)
	err = machine.RunAll()
	if err != nil {
		t.Error(err)
	}

	gBack := []float64{1, 1}
	grad := FromTensor(tf64.NewTensor(tf64.WithShape(x.shape...), tf64.WithBacking(gBack)))
	xG, _ := x.Grad()
	yG, _ := y.Grad()

	assert.Equal(grad, xG)
	assert.Equal(grad, yG)

	// tack more shit onto the graph, and execute it again
	szp2 := Must(Add(sz, twof64))
	szp3 := Must(Add(sz, threef64))

	var szp2Val Value
	readSzp2 := Read(szp2, &szp2Val)

	sg := g.SubgraphRoots(readSzp2, szp2)
	machine = NewLispMachine(sg)
	err = machine.RunAll()
	if err != nil {
		t.Error(err)
	}

	assert.NotNil(szp2Val)
	assert.Equal(szp2.Value(), szp2Val)
	assert.Nil(szp3.boundTo) // node that was not executed on should not have any values bound to it

	// play it again, sam!
	// this is to test that if given the same root that had previously been executed on, it will not reallocate a new *dv
	sg = g.SubgraphRoots(szp3)
	machine = NewLispMachine(sg)
	err = machine.RunAll()
	if err != nil {
		t.Error(err)
	}

	// save szp3's value
	szp3dv := szp3.boundTo.(*dualValue)
	szp3dvv := szp3dv.Value

	err = machine.RunAll()
	if err != nil {
		t.Error(err)
	}

	if dv := szp3.boundTo.(*dualValue); dv != szp3dv {
		t.Error("A new *dualValue had been allocated for szp3dv. That's not supposed to happen")
	} else if dv.Value != szp3dvv {
		t.Error("A new value for szp3dv.Value has been allocated. That ain't supposed to happen")
	}

	// idiotsville

	// non scalar costs
	cost := Must(Add(sz, x))
	sg = g.Subgraph(cost)
	machine = NewLispMachine(sg)
	err = machine.RunAll()
	if err == nil {
		t.Error("Expected a AutoDiff error")
	}
}

func TestLispMachineRepeatedRuns(t *testing.T) {
	assert := assert.New(t)
	var err error
	g := NewGraph()
	x := NewVector(g, Float64, WithShape(2), WithName("x"), WithInit(RangedFrom(0)))
	y := NewMatrix(g, Float64, WithShape(2, 3), WithName("y"), WithInit(RangedFrom(0)))
	z := Must(Mul(x, y))
	cost := Must(Slice(z, S(1))) // this simulates the more complex cost functions

	reps := 2

	for i := 0; i < reps; i++ {
		m := NewLispMachine(g)
		if err := m.RunAll(); err != nil {
			t.Errorf("Repetition %d error: %v", i, err)
			continue
		}

		var gradX, gradY, gradZ, gradC Value
		if gradX, err = x.Grad(); err != nil {
			t.Errorf("No gradient for x in repetition %d. Error: %v", i, err)
			continue
		}
		if gradY, err = y.Grad(); err != nil {
			t.Errorf("No gradient for y in repetition %d. Error: %v", i, err)
			continue
		}
		if gradZ, err = z.Grad(); err != nil {
			t.Errorf("No gradient for z in repetition %d. Error: %v", i, err)
			continue
		}
		if gradC, err = cost.Grad(); err != nil {
			t.Errorf("No gradient for cost in repetition %d. Error: %v", i, err)
			continue
		}

		assert.Equal([]float64{1, 4}, gradX.Data())
		assert.Equal([]float64{0, 0, 0, 0, 1, 0}, gradY.Data())
		assert.Equal([]float64{0, 1, 0}, gradZ.Data())
		assert.Equal(1.0, gradC.Data())

		// This simulates the cloberring of of the gradients of the nodes. The next iteration should STILL reveal the same results
		model := Nodes{x, y, z, cost}
		for _, n := range model {
			dv := n.boundTo.(*dualValue)
			if err = dv.SetDeriv(dv.d.zero()); err != nil {
				t.Errorf("Unable to set the gradient to 0 for %v. Error : %v", n, err)
				continue
			}
		}
	}

}
