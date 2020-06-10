package solvers

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
)

type DV struct {
	*dual.Dual
}

func (d DV) Value() values.Value { return d.Dual.Value }

func (d DV) Grad() (values.Value, error) {
	deriv := d.Dual.Deriv()
	if deriv == nil {
		return nil, errors.New("No Grad")
	}
	return deriv, nil
}

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

	dv := new(dual.Dual)
	dv.SetValue(v)
	dv.SetDeriv(d)

	n := DV{dv}

	model := []ValueGrad{n}
	return model
}

func tf32Node() []ValueGrad {
	backingV := []float32{1, 2, 3, 4}
	backingD := []float32{0.5, -10, 10, 0.5}

	v := tensor.New(tensor.WithBacking(backingV), tensor.WithShape(2, 2))
	d := tensor.New(tensor.WithBacking(backingD), tensor.WithShape(2, 2))

	dv := new(dual.Dual)
	dv.SetValue(v)
	dv.SetDeriv(d)

	n := DV{dv}
	model := []ValueGrad{n}
	return model
}

// The Rosenbrock function is a non-convex function,
// which is used as a performance test problem for optimization algorithms.
// https://en.wikipedia.org/wiki/Rosenbrock_function
//
// f(x,y) = (a-x)² + b(y-x²)²
// It has a global minimum at (x, y) = (a, a²), where f(x,y) = 0.
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
