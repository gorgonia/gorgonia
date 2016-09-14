package gorgonia

import (
	"io/ioutil"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestTapeVMBasic(t *testing.T) {

}

func TestTapeVMPutsGrads(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	x := NewScalar(g, Float64, WithName("x"))
	y := NewScalar(g, Float64, WithName("y"))

	res := Must(Mul(x, y))
	grads, err := Grad(res, x, y)
	if err != nil {
		t.Error(err)
		ioutil.WriteFile("fullGraph.dot", []byte(g.ToDot()), 0644)
	}

	prog, locMap, err := Compile(g)
	if err != nil {
		t.Error(err)
	}

	machine := NewTapeMachine(prog, locMap)
	machine.Let(x, 5.9)
	machine.Let(y, 3.1)
	err = machine.RunAll()

	if err != nil {
		t.Errorf("%v.\nProg:%v", prog)
	}

	xdv, ok := x.boundTo.(*dualValue)
	if !ok {
		t.Errorf("Expected node %v to have a boundTo that is a *dualValue", x)
	}

	ydv, ok := y.boundTo.(*dualValue)
	if !ok {
		t.Errorf("Expected node %v to have a boundTo that is a *dualValue", y)
	}

	assert.Equal(xdv.d, grads[0].boundTo)
	assert.Equal(ydv.d, grads[1].boundTo)
}
