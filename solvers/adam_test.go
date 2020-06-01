package solvers

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdamSolver(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dRosenbrock(1, 100, -0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.113
	if nil != err {
		t.Fatal(err)
	}

	solver := NewAdamSolver()

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
