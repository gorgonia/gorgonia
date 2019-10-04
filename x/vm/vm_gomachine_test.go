package xvm

import (
	"testing"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func TestChanDB(t *testing.T) {
	db := newChanDB()
	testChan := make(chan gorgonia.Value, 0)
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
	_ = NewGoMachine(gorgonia.NewGraph())
}

func TestGoMachine_RunAll(t *testing.T) {

	g := gorgonia.NewGraph()

	var x, y, z *gorgonia.Node

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	z, _ = gorgonia.Add(x, y)

	// create a VM to run the program on
	machine := NewGoMachine(g)
	defer machine.Close()

	// set initial values then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
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

	g := gorgonia.NewGraph()

	var x, y, z, zz *gorgonia.Node

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	z, _ = gorgonia.Add(x, y)
	zz, _ = gorgonia.Add(x, z)

	// create a VM to run the program on
	machine := NewGoMachine(g)
	defer machine.Close()

	// set initial values then run
	gorgonia.Let(x, 2.0)
	gorgonia.Let(y, 2.5)
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
	g := gorgonia.NewGraph()

	var x, y *gorgonia.Node

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	gorgonia.Add(x, y)
	edgesIT := g.Edges()
	if edgesIT.Len() != 2 {
		t.Fail()
	}
}
func TestGoMachine_MaxPool2D(t *testing.T) {
	dts := []tensor.Dtype{tensor.Float64, tensor.Float32}
	for _, dt := range dts {
		g := gorgonia.NewGraph()
		x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(1, 2, 3, 4), gorgonia.WithInit(gorgonia.RangedFrom(0)))
		_, err := gorgonia.MaxPool2D(x, tensor.Shape{2, 2}, []int{0, 0}, []int{1, 1})
		if err != nil {
			t.Fatal(err)
		}
		/*
			cost := Must(Sum(y))
				_, err = Grad(cost, x)
				if err != nil {
					t.Fatal(err)
				}
		*/

		m := NewGoMachine(g)
		if err := m.RunAll(); err != nil {
			t.Fatal(err)
		}
	}
}
