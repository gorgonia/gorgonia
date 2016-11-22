package gorgonia

import (
	"io/ioutil"
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/stretchr/testify/assert"
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

func TestBroadcast2(t *testing.T) {
	assert := assert.New(t)
	var g *ExprGraph
	var x, y, z *Node
	var m *lispMachine
	var err error

	xT := tf64.NewTensor(tf64.WithShape(2, 3), tf64.WithBacking(tf64.RangeFloat64(0, 6)))
	yT := tf64.NewTensor(tf64.WithShape(2, 1), tf64.WithBacking([]float64{100, 200}))

	g = NewGraph()
	x = NewMatrix(g, Float64, WithShape(2, 3), WithValue(xT), WithName("x"))
	y = NewVector(g, Float64, WithShape(2, 1), WithValue(yT), WithName("y"))
	z, err = Broadcast(addOpType, x, y, NewBroadcastPattern(nil, []byte{1}))
	if err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m = NewLispMachine(g, ExecuteFwdOnly())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	assert.Equal([]float64{100, 101, 102, 203, 204, 205}, extractF64s(z.Value()))

	g = NewGraph()
	x = NewMatrix(g, Float64, WithShape(2, 3), WithValue(xT), WithName("x"))
	y = NewVector(g, Float64, WithShape(2, 1), WithValue(yT), WithName("y"))
	z, err = Broadcast(addOpType, y, x, NewBroadcastPattern([]byte{1}, nil))
	if err != nil {
		ioutil.WriteFile("Broadcast.dot", []byte(g.ToDot()), 0644)
		t.Fatalf("%+v", err)
	}

	m = NewLispMachine(g, ExecuteFwdOnly())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	assert.Equal([]float64{100, 101, 102, 203, 204, 205}, extractF64s(z.Value()))

}
