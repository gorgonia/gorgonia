package gorgonia

import (
	"math"
	"testing"

	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/stretchr/testify/assert"
)

func clampFloat64(v, min, max float64) float64 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func TestRMSPropSolver(t *testing.T) {
	assert := assert.New(t)
	backingV := []float64{1, 2, 3, 4}
	backingD := []float64{0.5, -10, 10, 0.5}
	v := tf64.NewTensor(tf64.WithBacking(backingV), tf64.WithShape(2, 2))
	d := tf64.NewTensor(tf64.WithBacking(backingD), tf64.WithShape(2, 2))
	V := FromTensor(v)
	dv := dvUnit0(V)
	dv.d = FromTensor(d)

	n := new(Node)
	n.boundTo = dv

	model := Nodes{n}

	stepSize := 0.01
	l2Reg := 0.000001
	clip := 5.0

	s := NewRMSPropSolver(WithLearnRate(stepSize), WithL2Reg(l2Reg), WithClip(clip))

	correct := make([]float64, len(backingV))
	cached := make([]float64, len(backingV))
	for i := 0; i < 5; i++ {
		for j, v := range backingV {
			grad := backingD[j]
			cw := cached[j]

			decayed := cw*s.decay + (1.0-s.decay)*grad*grad
			cached[j] = decayed

			grad = clampFloat64(grad, -clip, clip)
			upd := -stepSize*grad/math.Sqrt(decayed+s.eps) - l2Reg*v
			correct[j] = v + upd
		}

		err := s.Step(model)
		if err != nil {
			t.Error(err)
		}

		sCache := s.cache[0].Value.(Tensor).Tensor.(*tf64.Tensor)
		assert.Equal(correct, backingV, "Iteration: %d", i)
		assert.Equal(cached, sCache.Data(), "Iteration: %d", i)

	}
}
