package gorgonia

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
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

func clampFloat32(v, min, max float32) float32 {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

func tf64Node() Nodes {
	backingV := []float64{1, 2, 3, 4}
	backingD := []float64{0.5, -10, 10, 0.5}
	v := tensor.New(tensor.WithBacking(backingV), tensor.WithShape(2, 2))
	d := tensor.New(tensor.WithBacking(backingD), tensor.WithShape(2, 2))

	dv := dvUnit0(v)
	dv.d = d

	n := new(Node)
	n.boundTo = dv

	model := Nodes{n}
	return model
}

func tf32Node() Nodes {
	backingV := []float32{1, 2, 3, 4}
	backingD := []float32{0.5, -10, 10, 0.5}

	v := tensor.New(tensor.WithBacking(backingV), tensor.WithShape(2, 2))
	d := tensor.New(tensor.WithBacking(backingD), tensor.WithShape(2, 2))

	dv := dvUnit0(v)
	dv.d = d

	n := new(Node)
	n.boundTo = dv

	model := Nodes{n}
	return model
}

func manualRMSProp64(t *testing.T, s *RMSPropSolver, model Nodes) {
	assert := assert.New(t)
	correct := make([]float64, 4)
	cached := make([]float64, 4)

	grad0, _ := model[0].Grad()
	backingV := model[0].Value().Data().([]float64)
	backingD := grad0.Data().([]float64)

	for i := 0; i < 5; i++ {
		for j, v := range backingV {
			grad := backingD[j]
			cw := cached[j]

			decayed := cw*s.decay + (1.0-s.decay)*grad*grad
			cached[j] = decayed

			grad = clampFloat64(grad, -s.clip, s.clip)
			upd := -s.eta*grad/math.Sqrt(decayed+s.eps) - s.l2reg*v
			correct[j] = v + upd
		}

		err := s.Step(model)
		if err != nil {
			t.Error(err)
		}

		sCache := s.cache[0].Value.(tensor.Tensor)
		assert.Equal(correct, backingV, "Iteration: %d", i)
		assert.Equal(cached, sCache.Data(), "Iteration: %d", i)

	}
}

func manualRMSProp32(t *testing.T, s *RMSPropSolver, model Nodes) {
	assert := assert.New(t)
	correct := make([]float32, 4)
	cached := make([]float32, 4)

	grad0, _ := model[0].Grad()
	backingV := model[0].Value().Data().([]float32)
	backingD := grad0.Data().([]float32)

	decay := float32(s.decay)
	l2reg := float32(s.l2reg)
	eta := float32(s.eta)
	eps := float32(s.eps)
	clip := float32(s.clip)

	for i := 0; i < 5; i++ {
		for j, v := range backingV {
			grad := backingD[j]
			cw := cached[j]

			decayed := cw*decay + (1.0-decay)*grad*grad
			cached[j] = decayed

			grad = clampFloat32(grad, -clip, clip)
			upd := -eta*grad/math32.Sqrt(decayed+eps) - l2reg*v
			correct[j] = v + upd
		}

		err := s.Step(model)
		if err != nil {
			t.Error(err)
		}

		sCache := s.cache[0].Value.(tensor.Tensor)
		assert.True(floatsEqual32(correct, backingV))
		assert.True(floatsEqual32(cached, sCache.Data().([]float32)))
	}
}

func TestRMSPropSolver(t *testing.T) {

	stepSize := 0.01
	l2Reg := 0.000001
	clip := 5.0

	var s *RMSPropSolver
	var model Nodes

	s = NewRMSPropSolver(WithLearnRate(stepSize), WithL2Reg(l2Reg), WithClip(clip))
	model = tf64Node()
	manualRMSProp64(t, s, model)

	s = NewRMSPropSolver(WithLearnRate(stepSize), WithL2Reg(l2Reg), WithClip(clip))
	model = tf32Node()
	manualRMSProp32(t, s, model)

}
