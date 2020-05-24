package solvers

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestAdaGradSolver(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dSquare(-0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.39
	if nil != err {
		t.Fatal(err)
	}

	solver := NewAdaGradSolver()

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
