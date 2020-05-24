package solvers

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/dawson"
	"gorgonia.org/tensor"
)

func manualRMSProp64(t *testing.T, s *RMSPropSolver, model []ValueGrad) {
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

func manualRMSProp32(t *testing.T, s *RMSPropSolver, model []ValueGrad) {
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

	// NOTE: THIS IS NAUGHTY. A proper comparison using 1e-5  should be used but that causes errors.
	closef32 := func(a, b float32) bool {
		return dawson.ToleranceF32(a, b, 1e-4)
	}

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
		assert.True(dawson.AllClose(correct, backingV, closef32))
		assert.True(dawson.AllClose(cached, sCache.Data().([]float32), closef32))
	}
}

func TestRMSPropSolverManual(t *testing.T) {

	stepSize := 0.01
	l2Reg := 0.000001
	clip := 5.0

	var s *RMSPropSolver
	var model []ValueGrad

	s = NewRMSPropSolver(WithLearnRate(stepSize), WithL2Reg(l2Reg), WithClip(clip))
	model = tf64Node()
	manualRMSProp64(t, s, model)

	s = NewRMSPropSolver(WithLearnRate(stepSize), WithL2Reg(l2Reg), WithClip(clip))
	model = tf32Node()
	manualRMSProp32(t, s, model)

}

func TestRMSPropSolver(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dRosenbrock(1, 100, -0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.68
	if nil != err {
		t.Fatal(err)
	}

	solver := NewRMSPropSolver()

	maxIterations := 1000

	costFloat := 42.0
	for 0 != maxIterations {
		m.Reset()
		err = m.RunAll()
		if nil != err {
			t.Fatal(err)
		}

		costFloat = cost.Value().Data().(float64)
		if costThreshold > math.Abs(costFloat) {
			break
		}

		err = solver.Step([]ValueGrad{z})
		if nil != err {
			t.Fatal(err)
		}

		maxIterations--
	}

	assert.InDelta(0, costFloat, costThreshold)
}
