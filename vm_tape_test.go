package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"
	"reflect"
	"strings"
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
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

	op.SetTraining(true)

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

type faultyOp struct{}

// Arity returns 1
func (i faultyOp) Arity() int { return 2 }

func (i faultyOp) Type() hm.Type {
	a := hm.TypeVariable('a')

	return hm.NewFnType(a, a, a)
}

func (i faultyOp) InferShape(ds ...DimSizer) (tensor.Shape, error) { return ds[0].(tensor.Shape), nil }

func (i faultyOp) Do(vs ...Value) (Value, error) {
	return vs[0], nil
}

func (i faultyOp) ReturnsPtr() bool      { return true }
func (i faultyOp) CallsExtern() bool     { return false }
func (i faultyOp) OverwritesInput() int  { return -1 }
func (i faultyOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "I") }

func (i faultyOp) Hashcode() uint32 {
	h := fnv.New32a()
	i.WriteHash(h)
	return h.Sum32()
}

func (i faultyOp) String() string { return "FaultyOp" }

func Test_tapeMachinePointerWatchOk(t *testing.T) {
	var err error

	c := require.New(t)

	g := NewGraph()

	a := NewTensor(g, Float32, 2, WithShape(2, 2), WithInit(GlorotN(1)), WithName("a"))

	out := Must(Mul(a, a))
	out = Must(Mul(a, out))
	out = Must(Mul(a, out))

	trainVM := NewTapeMachine(g, WithPointerWatch())

	for i := 0; i < 3; i++ {
		err = trainVM.RunAll()
		c.NoError(err)
	}
}

func Test_tapeMachinePointerWatchFail(t *testing.T) {
	var err error

	c := require.New(t)

	g := NewGraph()

	ts := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))

	a := NewTensor(g, Float32, 2, WithShape(2, 2), WithValue(ts), WithName("a"))
	b := NewTensor(g, Float32, 2, WithShape(2, 2), WithValue(ts), WithName("b"))

	prod := Must(HadamardProd(a, b))
	out := Must(ApplyOp(&faultyOp{}, prod, b))

	trainVM := NewTapeMachine(g, WithPointerWatch())
	err = trainVM.RunAll()
	c.Error(err)
	t.Logf("error: %v", err)

	// in this test, prod and out both have the same value which is not right
	t.Logf("prod: %v", prod.Value())
	t.Logf("out: %v", out.Value())

	c.True(strings.Contains(err.Error(), "Pointer clash found in value."))
}
