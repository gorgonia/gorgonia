package gorgonia

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_tapeMachine_Reset(t *testing.T) {
	g := NewGraph()

	var x, y, z *Node
	var err error

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	if z, err = Add(x, y); err != nil {
		t.Fatal(err)
	}

	// create a VM to run the program on
	m1 := NewTapeMachine(g)
	m2 := NewTapeMachine(g)
	defer m1.Close()
	defer m2.Close()

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err = m1.RunAll(); err != nil {
		t.Fatal(err)
	}
	if z.Value().Data().(float64) != 4.5 {
		t.Fatalf("Expected %v, got %v", 4.5, z.Value())
	}
	m1.Reset()
	if !reflect.DeepEqual(m1.locMap, m2.locMap) {
		t.Fatalf("expected locmap\n\n%#v, got\n\n%#v", m1, m2)
	}
	if !reflect.DeepEqual(m1.p, m2.p) {
		t.Fatalf("expected program\n\n%#v, got\n\n%#v", m1, m2)
	}
	if !reflect.DeepEqual(m1.cpumem, m2.cpumem) {
		t.Fatalf("expected cpumem\n\n%#v, got\n\n%#v", m1, m2)
	}
	if !reflect.DeepEqual(m1.gpumem, m2.gpumem) {
		t.Fatalf("expected gpumem\n\n%#v, got\n\n%#v", m1, m2)
	}
	if !reflect.DeepEqual(m1.pc, m2.pc) {
		t.Fatalf("expected pc\n\n%#v, got\n\n%#v", m1, m2)
	}
}

func Test_tapeMachineEvalMode(t *testing.T) {
	c := require.New(t)

	g := NewGraph()

	x := NewTensor(g, Float32, 2, WithShape(3, 2), WithInit(GlorotN(1)), WithName("x"))
	scale := NewTensor(g, Float32, 2, WithShape(1, 2), WithInit(GlorotN(1)), WithName("scale"))
	bias := NewTensor(g, Float32, 2, WithShape(1, 2), WithInit(GlorotN(1)), WithName("bias"))

	y, _, _, op, err := BatchNorm(x, scale, bias, 0.9, 1e-5)
	c.NoError(err)

	op.SetTraining()

	var yVal, scaleVal Value
	Read(y, &yVal)
	Read(scale, &scaleVal)

	cost, _ := Mean(y)

	if _, err := Grad(cost, x, scale, bias); err != nil {
		t.Fatal(err)
	}

	trainVM := NewTapeMachine(g, BindDualValues(x, scale, bias), TraceExec())

	err = trainVM.RunAll()
	c.NoError(err)

	evalVM := NewTapeMachine(g, EvalMode())
	err = evalVM.RunAll()
	c.NoError(err)
}
