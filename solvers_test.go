package gorgonia

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/stretchr/testify/assert"
	"gorgonia.org/dawson"
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

func tf64Node() []ValueGrad {
	backingV := []float64{1, 2, 3, 4}
	backingD := []float64{0.5, -10, 10, 0.5}
	v := tensor.New(tensor.WithBacking(backingV), tensor.WithShape(2, 2))
	d := tensor.New(tensor.WithBacking(backingD), tensor.WithShape(2, 2))

	dv := dvUnit0(v)
	dv.d = d

	n := new(Node)
	n.boundTo = dv

	model := []ValueGrad{n}
	return model
}

func tf32Node() []ValueGrad {
	backingV := []float32{1, 2, 3, 4}
	backingD := []float32{0.5, -10, 10, 0.5}

	v := tensor.New(tensor.WithBacking(backingV), tensor.WithShape(2, 2))
	d := tensor.New(tensor.WithBacking(backingD), tensor.WithShape(2, 2))

	dv := dvUnit0(v)
	dv.d = d

	n := new(Node)
	n.boundTo = dv

	model := []ValueGrad{n}
	return model
}

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

func TestVanillaSolver(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dRosenbrock(1, 100, -0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.185
	if nil != err {
		t.Fatal(err)
	}

	solver := NewVanillaSolver()

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

func TestMomentum(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dRosenbrock(1, 100, -0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.39
	if nil != err {
		t.Fatal(err)
	}

	solver := NewMomentum()

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

func TestBarzilaiBorweinSolver(t *testing.T) {
	assert := assert.New(t)

	z, cost, m, err := model2dRosenbrock(1, 100, -0.5, 0.5)
	defer m.Close()
	const costThreshold = 0.00002
	if nil != err {
		t.Fatal(err)
	}

	solver := NewBarzilaiBorweinSolver(WithLearnRate(0.0001))

	maxIterations := 200

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

// The Rosenbrock function is a non-convex function,
// which is used as a performance test problem for optimization algorithms.
// https://en.wikipedia.org/wiki/Rosenbrock_function
//
// f(x,y) = (a-x)^2 + b(y-x^2)^2
// It has a global minimum at (x, y) = (a, a^2), where f(x,y) = 0.
// Usually a = 1, b = 100, then the minimum is at x = y = 1
// TODO: There is also an n-dimensional version...see wiki
func model2dRosenbrock(a, b, xInit, yInit float64) (z, cost *Node, machine *tapeMachine, err error) {
	g := NewGraph()

	z = NewTensor(g, Float64, 1, WithShape(2), WithName("z"))

	aN := NewConstant(a, WithName("a"))
	bN := NewConstant(b, WithName("b"))

	xProjFloat := []float64{1, 0}
	xProj := NewConstant(tensor.New(tensor.WithBacking(xProjFloat), tensor.WithShape(2)))

	yProjFloat := []float64{0, 1}
	yProj := NewConstant(tensor.New(tensor.WithBacking(yProjFloat), tensor.WithShape(2)))

	x := Must(Mul(z, xProj))
	y := Must(Mul(z, yProj))

	// First term

	sqrt1stTerm := Must(Sub(aN, x))

	firstTerm := Must(Square(sqrt1stTerm))

	// Second term

	xSquared := Must(Square(x))

	yMinusxSquared := Must(Sub(y, xSquared))

	yMinusxSquaredSqu := Must(Square(yMinusxSquared))

	secondTerm := Must(Mul(bN, yMinusxSquaredSqu))

	// cost
	cost = Must(Add(firstTerm, secondTerm))

	dcost, err := Grad(cost, z)
	if nil != err {
		return nil, nil, nil, err
	}

	prog, locMap, err := CompileFunction(g, Nodes{z}, Nodes{cost, dcost[0]})
	if nil != err {
		return nil, nil, nil, err
	}

	machine = NewTapeMachine(g, WithPrecompiled(prog, locMap), BindDualValues(z))

	err = machine.Let(z, tensor.New(tensor.WithBacking([]float64{xInit, yInit}), tensor.WithShape(2)))
	if nil != err {
		return nil, nil, nil, err
	}

	return
}

func model2dSquare(xInit, yInit float64) (z, cost *Node, machine *tapeMachine, err error) {
	g := NewGraph()

	z = NewTensor(g, Float64, 1, WithShape(2), WithName("z"))

	// cost
	cost = Must(Mul(z, z))

	dcost, err := Grad(cost, z)
	if nil != err {
		return nil, nil, nil, err
	}

	prog, locMap, err := CompileFunction(g, Nodes{z}, Nodes{cost, dcost[0]})
	if nil != err {
		return nil, nil, nil, err
	}

	machine = NewTapeMachine(g, WithPrecompiled(prog, locMap), BindDualValues(z))

	err = machine.Let(z, tensor.New(tensor.WithBacking([]float64{xInit, yInit}), tensor.WithShape(2)))
	if nil != err {
		return nil, nil, nil, err
	}

	return
}
