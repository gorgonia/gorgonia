package gorgonia

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func TestLift2Broadcast(t *testing.T) {
	lifted := Lift2Broadcast(BroadcastAdd)
	t.Run("first input error short-circuits", func(t *testing.T) {
		g := NewGraph()
		b := NewMatrix(g, Float64, WithShape(2, 2), WithValue(
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		))
		res := lifted(Err(errors.New("bad a")), b, nil, nil)
		assert.Error(t, res.Err())
		assert.Nil(t, res.Node())
	})

	t.Run("second input error short-circuits", func(t *testing.T) {
		g := NewGraph()
		a := NewVector(g, Float64, WithShape(2), WithValue(
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		))
		res := lifted(a, Err(errors.New("bad b")), nil, nil)
		assert.Error(t, res.Err())
		assert.Nil(t, res.Node())
	})

	t.Run("successful lift returns executable result", func(t *testing.T) {
		g := NewGraph()
		a := NewVector(g, Float64, WithShape(2), WithValue(
			tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{10, 20})),
		))
		b := NewMatrix(g, Float64, WithShape(2, 2), WithValue(
			tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{1, 2, 3, 4})),
		))
		res := lifted(a, b, []byte{1}, nil)
		assert.NoError(t, res.Err())
		assert.NotNil(t, res.Node())

		m := NewLispMachine(g, ExecuteFwdOnly())
		defer m.Close()
		if err := m.RunAll(); err != nil {
			t.Fatalf("%+v", err)
		}
		assert.Equal(t, []float64{11, 12, 23, 24}, extractF64s(res.Node().Value()))
	})
}
