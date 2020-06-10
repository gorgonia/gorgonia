package exprgraph

import (
	"testing"

	"gorgonia.org/tensor"
)

func TestRepeatedInsert(t *testing.T) {
	g := New(tensor.StdEng{})

	x := Make(g, "x", tensor.WithShape(), tensor.Of(tensor.Float64))
	for i := 0; i < 100; i++ {
		g.Insert(x)
	}

	if len(g.nodes) != 1 {
		t.Errorf("Expected only one node in graph")
	}
}
