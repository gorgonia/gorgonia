package gorgonia

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestBroadcast_CNHW(t *testing.T) {
	if CUDA {
		t.SkipNow()
	}

	assert := assert.New(t)
	var g *ExprGraph
	var x, y, z *Node
	var m *lispMachine
	var err error

	xT := tensor.New(tensor.WithShape(1, 2, 3, 3), tensor.WithBacking(tensor.Range(tensor.Float64, 0, 18)))
	yT := tensor.New(tensor.WithShape(1, 2, 1, 1), tensor.WithBacking([]float64{100, 200}))
	t.Log(xT)

	g = NewGraph()
	x = NewTensor(g, Float64, 4, WithValue(xT), WithName("x"))
	y = NewTensor(g, Float64, 4, WithValue(yT), WithName("y"))
	if z, err = Add(x, y, NewBroadcastPattern(nil, []byte{2, 3})); err != nil {
		ioutil.WriteFile("Broadcast_CNHW.dot", []byte(g.ToDot()), 0644)
		t.Fatal(err)
	}

	m = NewLispMachine(g, ExecuteFwdOnly())
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	t.Log(z.Value())
	assert.Equal([]float64{
		100, 101, 102,
		103, 104, 105,
		106, 107, 108,
		209, 210, 211,
		212, 213, 214,
		215, 216, 217,
	}, extractF64s(z.Value()))
}
