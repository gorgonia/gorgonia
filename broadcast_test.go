package gorgonia

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestBroadcastPattern(t *testing.T) {
	assert := assert.New(t)
	var bcpat BroadcastPattern

	// make sure that the basics work
	bcpat = NewBroadcastPattern(nil, []byte{1})
	assert.Equal(BroadcastPattern(0x02), bcpat)

	bcpat = NewBroadcastPattern(nil, []byte{0})
	assert.Equal(BroadcastPattern(0x01), bcpat)

	bcpat = NewBroadcastPattern([]byte{1, 0}, nil)
	assert.Equal(BroadcastPattern(0x30), bcpat)

	bcpat = NewBroadcastPattern([]byte{0}, nil)
	assert.Equal(BroadcastPattern(0x10), bcpat)

	// checks
	bcpat = NewBroadcastPattern(nil, []byte{1})
	assert.True(bcpat.bc(false, 1))
	assert.False(bcpat.bc(true, 1))

	// ons
	bcpat = NewBroadcastPattern(nil, []byte{1})
	assert.Equal([]int{1}, bcpat.on()[1])
	assert.Nil(bcpat.on()[0])

	bcpat = NewBroadcastPattern([]byte{2, 1}, []byte{1})
	assert.Equal([]int{1, 2}, bcpat.on()[0])
	assert.Equal([]int{1}, bcpat.on()[1])
}

func TestBroadcast(t *testing.T) {
	if CUDA {
		t.SkipNow()
	}

	assert := assert.New(t)
	var g *ExprGraph
	var x, y, a, b, z *Node
	var m *lispMachine
	var err error

	xT := tensor.New(tensor.WithShape(2, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 6)))
	yT := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{100, 200}))

	g = NewGraph()
	x = NewMatrix(g, Float64, WithShape(2, 3), WithValue(xT), WithName("x"))
	y = NewVector(g, Float64, WithShape(2), WithValue(yT), WithName("y"))
	if a, b, err = Broadcast(x, y, NewBroadcastPattern(nil, []byte{1})); err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}
	z, err = Add(a, b)
	if err != nil {
		t.Fatalf("Error: %v. a %v + b %v", err, a.Shape(), b.Shape())
	}
	if _, _, err = Broadcast(x, y, NewBroadcastPattern(nil, []byte{1})); err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m = NewLispMachine(g, ExecuteFwdOnly())
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	assert.Equal([]float64{100, 101, 102, 203, 204, 205}, extractF64s(z.Value()))

	g = NewGraph()
	x = NewMatrix(g, Float64, WithShape(2, 3), WithValue(xT), WithName("x"))
	y = NewVector(g, Float64, WithShape(2), WithValue(yT), WithName("y"))
	if a, b, err = Broadcast(y, x, NewBroadcastPattern([]byte{1}, nil)); err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatalf("%+v", err)
	}
	// TODO: Check the error returned by Add?
	z, _ = Add(a, b)
	if _, _, err = Broadcast(x, y, NewBroadcastPattern(nil, []byte{1})); err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m = NewLispMachine(g, ExecuteFwdOnly())
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	assert.Equal([]float64{100, 101, 102, 203, 204, 205}, extractF64s(z.Value()))
}
