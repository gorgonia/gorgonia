package gorgonia

import (
	"io/ioutil"
	"testing"

	"gorgonia.org/tensor"
)

func TestBatchNormUnderstanding(t *testing.T) {
	g := NewGraph()

	xV := tensor.New(tensor.WithShape(2, 3, 2, 1), tensor.WithBacking([]float64{
		0.0001,
		0.003,

		0.02,
		0.007,

		0.007,
		0.05,

		// ---

		0.00015,
		0.0035,

		0.025,
		0.0075,

		0.0075,
		0.055,
	}))
	x := NodeFromAny(g, xV, WithName("x"))

	s2 := NewTensor(g, x.Dtype(), x.Shape().Dims(), WithShape(x.Shape().Clone()...), WithInit(Ones()), WithName("Scale"))
	b2 := NewTensor(g, x.Dtype(), x.Shape().Dims(), WithShape(x.Shape().Clone()...), WithInit(Zeroes()), WithName("Bias"))

	y2, scale2, bias2, op2, err := BatchNorm(x, s2, b2, 0.1, 1e-05)
	if err != nil {
		t.Fatal(err)
	}

	WithName("y2")(y2)

	op2.SetTraining()

	C, _ := Sum(y2)

	_, err = Grad(C, x, scale2, bias2)
	if err != nil {
		t.Fatal(err)
	}

	m := NewTapeMachine(g, TraceExec())
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	ioutil.WriteFile("foo.dot", []byte(g.ToDot()), 0644)
	t.Logf("\n%v", y2.Value())
	// _, _ = scale, scale2
	// _, _ = bias, bias2
	// _, _ = op, op2

}
