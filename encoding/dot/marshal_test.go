package dot

import (
	"testing"

	"gorgonia.org/gorgonia"
)

func TestMarshal(t *testing.T) {
	g := gorgonia.NewGraph()

	var x, y *gorgonia.Node
	var err error

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	if _, err = gorgonia.Add(x, y); err != nil {
		t.Fatal(err)
	}
	b, err := Marshal(g)
	if err != nil {
		t.Fatal(err)
	}
	_ = b
}
