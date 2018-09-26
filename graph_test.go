package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
	"gonum.org/v1/gonum/graph/topo"
	"gorgonia.org/tensor"
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
	assert.Equal(correctTo, graphNodeToNode(g.To(y.ID())))
	assert.Equal(correctTo, graphNodeToNode(g.To(x.ID())))

	assert.Equal(3, g.Nodes().Len())

	// Now, time to deal with constants
	xy1 := Must(Add(xy, onef64))
	assert.Nil(onef64.g)
	assert.Equal(g, xy1.g)

	var containsOne bool
	it := g.Nodes()
	for it.Next() {
		node := it.Node()
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
	t.Logf("%v", graphNodeToNode(iterator.NewOrderedNodes(sortedNodes)))
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
	assert.True(g.Has(col.ID()))

	colleen := new(Node)
	colleen.name = "COLLEEN THE COLLISION"
	colleen.hash = 0xdeadbeef
	colleen.hashed = true
	colleen2 := g.AddNode(colleen)

	assert.Equal(colleen, colleen2)
	assert.Equal(5, len(g.AllNodes()), "%v", g.AllNodes())
	assert.True(g.Has(colleen.ID()))

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
	g2 := g.subgraph(sub, true)

	t.Logf("%v", g2.AllNodes())

	if sortedNodes, err = Sort(g2); err != nil {
		t.Fatal(err)
	}
	assert.NotContains(sortedNodes, z)
	assert.Contains(g2.roots, x)
	assert.Contains(g2.roots, y)
	assert.Equal(2, len(g2.roots))
}

func TestGraph_SubgraphRoots(t *testing.T) {
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

func TestGraph_ExactSubgraphRoots(t *testing.T) {
	assert := assert.New(t)
	g, x, y, z := simpleVecEqn()
	sz := Must(Sum(z))
	setXtoZ := Set(x, z) // setting x = z

	sg0 := g.SubgraphRoots(sz)
	sg1 := g.ExactSubgraphRoots(sz)
	ns0 := sg0.AllNodes()
	ns1 := sg1.AllNodes()
	assert.Contains(ns0, setXtoZ)
	assert.NotContains(ns1, setXtoZ)
	assert.Contains(ns0, x)
	assert.Contains(ns0, y)
	assert.Contains(ns0, z)
	assert.Contains(ns0, sz)

}

func TestGraph_Constant(t *testing.T) {
	g := NewGraph()

	v1 := newF64(1.0)
	c0 := g.Constant(v1)
	c1 := g.Constant(v1)

	if c0 != c1 {
		t.Errorf("Expected c0 and c1 to be the same (pointer and all that)")
	}
}

func TestGraph_Clone(t *testing.T) {
	g, x, y, z := simpleVecEqn()
	z2 := Must(Square(z))

	// add a collided
	z2t := z2.Type()
	delete(g.byHash, z2.hash)
	g.byHash[0xdeadbeef] = z2
	col := new(Node)
	col.g = g
	col.name = "COLIN THE COLLISION"
	col.hash = 0xdeadbeef
	col.hashed = true
	col.boundTo = newF64(0)
	col.t = z2t
	g.AddNode(col)

	colleen := new(Node)
	colleen.g = g
	colleen.name = "COLLEEN THE COLLISION"
	colleen.hash = 0xdeadbeef
	colleen.hashed = true
	colleen.boundTo = newF64(0)
	colleen.t = z2t
	g.AddNode(colleen)

	one := onef64
	z2p1 := Must(Add(z2, one))                                    // add a constant
	rando := UniformRandomNode(g, Float64, 0, 1, z2p1.Shape()...) // add a weird node
	blah := Must(HadamardProd(z2p1, rando))
	cost := Must(Sum(blah))
	_, err := Grad(cost, x, y)
	if err != nil {
		t.Fatal(err)
	}

	g.Roots() // call it to populate the roots field

	// clone with nil values
	g2 := g.Clone().(*ExprGraph)
	for i, n := range g.all {
		cloned := g2.all[i]
		if !deepNodeEq(n, cloned) {
			t.Errorf("Expected %d of all to be %v. Got %v instead", i, n, cloned)
			break
		}
	}
	if len(g.evac) != len(g2.evac) && len(g.evac) > 0 {
		t.Errorf("Expected the evacs to have the same length")
	}
	for k, v := range g.evac {
		var v2 Nodes
		var ok bool
		if v2, ok = g2.evac[k]; !ok {
			t.Errorf("Key %v not found in cloned evac", k)
			break
		}
		for i, n := range v {
			if !deepNodeEq(n, v2[i]) {
				t.Errorf("Expected v[%d] to have equal values", i)
				break
			}
		}
		if t.Failed() {
			break
		}
	}
	if len(g.roots) != len(g2.roots) {
		t.Errorf("Expected roots to be %d. Got %d instead", len(g.roots), len(g2.roots))
	}
	for i, root := range g.roots {
		if !deepNodeEq(root, g2.roots[i]) {
			t.Errorf("Expected roots[%d] to have equal nodes", i)
			break
		}
	}

	if len(g.leaves) != len(g2.leaves) {
		t.Errorf("Expected leaves to be %d. Got %d instead", len(g.leaves), len(g2.leaves))
	}
	for i, leaf := range g.leaves {
		if !deepNodeEq(leaf, g2.leaves[i]) {
			t.Errorf("Expected leaves[%d] to be equal", i)
			break
		}
	}

	Let(x, tensor.New(tensor.WithBacking([]float64{1, 2})))
	Let(y, tensor.New(tensor.WithBacking([]float64{3, 4})))
	m := NewLispMachine(g, ExecuteFwdOnly()) // the gradient has been precalculated
	defer m.Close()
	if err := m.RunAll(); err != nil {
		t.Fatal(err)
	}

	g2 = g.Clone().(*ExprGraph)
	for i, n := range g.all {
		cloned := g2.all[i]
		if !deepNodeEq(n, cloned) {
			t.Errorf("Expected %d of all to be %v. Got %v instead", i, n, cloned)
			break
		}
	}
	if len(g.evac) != len(g2.evac) && len(g.evac) > 0 {
		t.Errorf("Expected the evacs to have the same length")
	}
	for k, v := range g.evac {
		var v2 Nodes
		var ok bool
		if v2, ok = g2.evac[k]; !ok {
			t.Errorf("Key %v not found in cloned evac", k)
			break
		}
		for i, n := range v {
			if !deepNodeEq(n, v2[i]) {
				t.Errorf("Expected v[%d] to have equal values", i)
				break
			}
		}
		if t.Failed() {
			break
		}
	}
	if len(g.roots) != len(g2.roots) {
		t.Errorf("Expected roots to be %d. Got %d instead", len(g.roots), len(g2.roots))
	}
	for i, root := range g.roots {
		if !deepNodeEq(root, g2.roots[i]) {
			t.Errorf("Expected roots[%d] to have equal nodes", i)
			break
		}
	}

	if len(g.leaves) != len(g2.leaves) {
		t.Errorf("Expected leaves to be %d. Got %d instead", len(g.leaves), len(g2.leaves))
	}
	for i, leaf := range g.leaves {
		if !deepNodeEq(leaf, g2.leaves[i]) {
			t.Errorf("Expected leaves[%d] to be equal", i)
			break
		}
	}
}
