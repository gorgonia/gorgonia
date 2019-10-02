package gorgonia

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestChanDB(t *testing.T) {
	db := newChanDB()
	testChan := make(chan Value, 0)
	db.upsert(testChan, 0, 0)
	_, ok := db.getChan(1, 1)
	if ok == true {
		t.Fail()
	}
	outputChan, ok := db.getChan(0, 0)
	if ok == false {
		t.Fail()
	}
	if outputChan != testChan {
		t.Fail()
	}
}
func TestNewGoMachine(t *testing.T) {
	_ = NewGoMachine(NewGraph())
}

func TestGoMachine_RunAll(t *testing.T) {

	g := NewGraph()

	var x, y, z *Node

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z, _ = Add(x, y)

	// create a VM to run the program on
	machine := NewGoMachine(g)
	defer machine.Close()

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	if z.Value() == nil {
		t.Fatal("z's value is nil")
	}
	if z.Value().Data().(float64) != float64(4.5) {
		t.Fail()
	}
}
func TestGoMachine_RunAllComplex(t *testing.T) {

	g := NewGraph()

	var x, y, z, zz *Node

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	z, _ = Add(x, y)
	zz, _ = Add(x, z)

	// create a VM to run the program on
	machine := NewGoMachine(g)
	defer machine.Close()

	// set initial values then run
	Let(x, 2.0)
	Let(y, 2.5)
	if err := machine.RunAll(); err != nil {
		t.Fatal(err)
	}

	if zz.Value() == nil {
		t.Fatal("z's value is nil")
	}
	if zz.Value().Data().(float64) != float64(6.5) {
		t.Fail()
	}
}

func TestGetEdges(t *testing.T) {
	g := NewGraph()

	var x, y *Node

	// define the expression
	x = NewScalar(g, Float64, WithName("x"))
	y = NewScalar(g, Float64, WithName("y"))
	Add(x, y)
	edgesIT := getEdges(g)
	if edgesIT.Len() != 2 {
		t.Fail()
	}
}
func TestGoMachine_MaxPool2D(t *testing.T) {
	dts := []tensor.Dtype{tensor.Float64, tensor.Float32}
	for _, dt := range dts {
		g := NewGraph()
		x := NewTensor(g, dt, 4, WithShape(1, 2, 3, 4), WithInit(RangedFrom(0)))
		y, err := MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		cost := Must(Sum(y))
		_, err = Grad(cost, x)
		if err != nil {
			t.Fatal(err)
		}

		m := NewGoMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatal(err)
		}
	}
}
