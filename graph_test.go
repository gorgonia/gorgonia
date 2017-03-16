package gorgonia

import (
	"testing"

	"github.com/gonum/graph"
	"github.com/gonum/graph/topo"
	"github.com/stretchr/testify/assert"
)

func TestGraphBasics(t *testing.T) {
	assert := assert.New(t)
	g, x, y, xy := simpleEqn()

	// basic stuff
	assert.Equal(g, xy.g)
	assert.Contains(g.AllNodes(), x)
	assert.Contains(g.AllNodes(), y)
	assert.Contains(g.AllNodes(), xy)

	assert.Equal(Nodes{x, y}, g.leaves)

	// Node/addressing stuff
	xid := x.ID()
	xFromID := g.Node(xid)
	assert.Equal(x, xFromID)

	var correctTo Nodes
	correctTo = Nodes{xy}
	assert.Equal(correctTo, g.to[x])
	assert.Equal(correctTo, g.to[y])

	// test Uniquifying ability of ExprGraph
	newX := g.AddNode(x)
	assert.Equal(x, newX)

	newY := g.AddNode(y)
	assert.Equal(y, newY)

	newXY := Must(Add(x, y))
	correctTo = append(correctTo, xy) // note this is correct. .Set() will be called when graph.To() is called
	assert.Equal(xy, newXY)
	assert.Equal(correctTo, g.to[y])
	assert.Equal(correctTo, g.to[x])

	correctTo = Nodes{xy}
	assert.Equal(correctTo, graphNodeToNode(g.To(y)))
	assert.Equal(correctTo, graphNodeToNode(g.To(x)))

	assert.Equal(3, len(g.Nodes()))

	// Now, time to deal with constants
	xy1 := Must(Add(xy, onef64))
	assert.Nil(onef64.g)
	assert.Equal(g, xy1.g)

	var containsOne bool
	for _, node := range g.Nodes() {
		n := node.(*Node)
		if n.Hashcode() == onef64.Hashcode() {
			containsOne = true
			break
		}
	}
	if !containsOne {
		t.Errorf("graph does not contain a clone of onef64: %v", g.Nodes())
	}

	// duplicate constants
	one := NewConstant(1.0)
	newOne := g.AddNode(one)
	if one == newOne {
		t.Error("one should not have been added to the graph")
	}
	assert.NotNil(newOne.g)
	assert.NotEqual(one, newOne)
}

// This test is added to make sure I'm sane when dealing with sorted graphs
// because sometimes Eobard Thawne is needed
func TestGraphSort(t *testing.T) {
	assert := assert.New(t)
	g, _, _, z := simpleVecEqn()
	WithName("z")(z)

	var sortedNodes []graph.Node
	var err error

	// stability tests
	for i := 0; i < 100; i++ {
		if sortedNodes, err = topo.Sort(g); err != nil {
			t.Error(err)
		}
		// expected := Nodes{z, y, x} // the old version of ExprGraph was stable with topo.Sort, but the new version ain't
		// assert.Equal(expected, sortedNodes)
		assert.Equal(z, sortedNodes[0])
	}

	// this is to remind myself how this thing sorts:
	t.Logf("%v", graphNodeToNode(sortedNodes))
}

// test that collisions are handled correctly
func TestGraphCollisions(t *testing.T) {
	assert := assert.New(t)
	g, _, _, xy := simpleEqn()
	delete(g.byHash, xy.hash)
	g.byHash[0xdeadbeef] = xy
	xy.hash = 0xdeadbeef
	xy.name = "original"
	t.Logf("original: %p, hash %x", xy, xy.Hashcode())

	col := new(Node)
	col.name = "COLIN THE COLLISION"
	col.hash = 0xdeadbeef
	col.hashed = true
	col2 := g.AddNode(col)

	assert.Equal(col, col2)
	assert.Equal(4, len(g.AllNodes()), "%v", g.AllNodes())
	assert.True(g.Has(col))

	colleen := new(Node)
	colleen.name = "COLLEEN THE COLLISION"
	colleen.hash = 0xdeadbeef
	colleen.hashed = true
	colleen2 := g.AddNode(colleen)

	assert.Equal(colleen, colleen2)
	assert.Equal(5, len(g.AllNodes()), "%v", g.AllNodes())
	assert.True(g.Has(colleen))

}

func TestGraphEquality(t *testing.T) {
	_, x, y, z := simpleVecEqn()

	xh1 := x.Hashcode()
	yh1 := y.Hashcode()
	if xh1 == yh1 {
		t.Error("Different nodes, should have different hashes")
	}

	_, x2, y2, z2 := simpleVecEqn()

	if x.Hashcode() != x2.Hashcode() {
		t.Error("They should have the same hash")
	}

	if y.Hashcode() != y2.Hashcode() {
		t.Error("They should have the same hash")
	}

	if z.Hashcode() != z2.Hashcode() {
		t.Error("They should have the same hash")
	}
}

func TestGraphSubgraph(t *testing.T) {
	var err error
	var sortedNodes Nodes
	assert := assert.New(t)

	g, x, y, z := simpleVecEqn()

	sub := Nodes{x, y}
	g2 := g.subgraph(sub)

	t.Logf("%v", g2.AllNodes())

	if sortedNodes, err = Sort(g2); err != nil {
		t.Fatal(err)
	}
	assert.NotContains(sortedNodes, z)
	assert.Contains(g2.roots, x)
	assert.Contains(g2.roots, y)
	assert.Equal(2, len(g2.roots))
}

func TestGraphSubgraphRoots(t *testing.T) {
	assert := assert.New(t)
	g, x, y, z := simpleVecEqn()
	sz := Must(Sum(z))
	a := NewVector(g, Float64, WithName("a"), WithShape(2))
	b := NewVector(g, Float64, WithName("b"), WithShape(2))
	c := Must(Add(a, b))
	sc := Must(Sum(c))

	var szVal, scVal Value
	readSZ := Read(sz, &szVal)
	readSC := Read(sc, &scVal)

	// check that stmt nodes aren't included in the roots
	sg := g.SubgraphRoots(readSZ, readSC)
	assert.Contains(sg.roots, sz)
	assert.Contains(sg.roots, sc)
	assert.Equal(2, len(sg.roots))

	// check that subgrapphing actually works
	sg = g.SubgraphRoots(c)
	ns := sg.AllNodes()
	assert.NotContains(ns, sc)
	assert.NotContains(ns, readSC)
	assert.NotContains(ns, x)
	assert.NotContains(ns, y)
	assert.NotContains(ns, z)
	assert.NotContains(ns, sz)
	assert.NotContains(ns, readSZ)
}
