package gorgonia

import (
	"reflect"
	"testing"
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
