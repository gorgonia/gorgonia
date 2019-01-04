package gorgonia

import (
	"reflect"
	"testing"

	"gorgonia.org/tensor"
)

var stdengType reflect.Type

func init() {
	stdengType = reflect.TypeOf(StandardEngine{})
}

func assertEngine(v value.Value, eT reflect.Type) bool {
	te := engineOf(v)
	if te == nil {
		return true
	}
	teT := reflect.TypeOf(te)
	return eT == teT
}

func assertGraphEngine(t *testing.T, g *ExprGraph, eT reflect.Type) {
	for _, n := range g.AllNodes() {
		if n.isInput() {
			inputEng := reflect.TypeOf(engineOf(n.Value()))
			if grad, err := n.Grad(); err == nil {
				if !assertEngine(grad, inputEng) {
					t.Errorf("Expected input %v value and gradient to share the same engine %v: Got %T", n.Name(), inputEng, engineOf(grad))
					return
				}
			}
			continue
		}
		if !assertEngine(n.Value(), eT) {
			t.Errorf("Expected node %v to be %v. Got %T instead", n, eT, engineOf(n.Value()))
			return
		}

		if grad, err := n.Grad(); err == nil {
			if !assertEngine(grad, eT) {
				t.Errorf("Expected gradient of node %v to be %v. Got %T instead", n, eT, engineOf(grad))
				return
			}
		}
	}
}

func engineOf(v value.Value) tensor.Engine {
	if t, ok := v.(tensor.Tensor); ok {
		return t.Engine()
	}
	return nil
}

func TestBasicEngine(t *testing.T) {
	g, x, y, _ := simpleVecEqn()

	Let(x, tensor.New(tensor.WithBacking([]float64{0, 1})))
	Let(y, tensor.New(tensor.WithBacking([]float64{3, 2})))
	m := NewTapeMachine(g, TraceExec())
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}
	if assertGraphEngine(t, g, stdengType); t.Failed() {
		t.FailNow()
	}
}
