package dot

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia"
)

// unmangleName replaces the pointer-based name with the node name.
//
// Node names are unique in Gorgonia graphs.
// However we cannot use node names in a `.dot` file because there are some names
// that will not be accepted by graphviz.
//
// Hence we use pointers as part of the name of the nodes. A node will have a name
// like "Node_0xbadb100d" (shout out to Tay Tay!) in the `.dot` file.
//
// unmangleName replaces `Node_0xbadc0ffee` with the name of the node for the purposes of testing.
func unmangleName(marshalled string, node *gorgonia.Node) string {
	name := node.Name()
	ptrName := fmt.Sprintf("%p", node)

	return strings.Replace(marshalled, ptrName, name, -1)
}

func TestIssue_407(t *testing.T) {
	// originally written by @wzzhu

	assert := assert.New(t)
	g := gorgonia.NewGraph()
	var x, y, z *gorgonia.Node
	var err error

	// define the expression
	x = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("x"))
	y = gorgonia.NewScalar(g, gorgonia.Float64, gorgonia.WithName("y"))
	if z, err = gorgonia.Add(x, y); err != nil {
		t.Fatal(err)
	}
	b, err := Marshal(g)
	if err != nil {
		t.Errorf("Error printing graph: %v", err)
	}
	fst := fmt.Sprintf("\n%s", string(b))
	fst = unmangleName(fst, x)
	fst = unmangleName(fst, y)
	fst = unmangleName(fst, z)

	var x2, y2, z2 *gorgonia.Node

	g2 := gorgonia.NewGraph()
	x2 = gorgonia.NewScalar(g2, gorgonia.Float64, gorgonia.WithName("x"))
	y2 = gorgonia.NewScalar(g2, gorgonia.Float64, gorgonia.WithName("y"))
	if z2, err = gorgonia.Add(x2, y2); err != nil {
		t.Fatal(err)
	}
	b2, err := Marshal(g2)
	if err != nil {
		t.Errorf("Error printing graph: %v", err)
	}
	snd := fmt.Sprintf("\n%s", string(b2))
	snd = unmangleName(snd, x2)
	snd = unmangleName(snd, y2)
	snd = unmangleName(snd, z2)
	assert.Equal(fst, snd, "XXX")
	t.Logf("%v %v", z, z2)
}

func TestStressTest407(t *testing.T) {
	for i := 0; i < 1024; i++ {
		TestIssue_407(t)
		if t.Failed() {
			t.Errorf("Failed at iteration %d", i)
			break
		}
	}
}
