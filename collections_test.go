package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSet(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	n0 := newNodeFromPool(withGraph(g), WithName("n0"))
	n1 := newNodeFromPool(withGraph(g), WithName("n1"))
	n2 := newNodeFromPool(withGraph(g), WithName("n2"))
	n3 := newNodeFromPool(withGraph(g), WithName("n3"))

	// calculate hashcode first
	n0.Hashcode()
	n1.Hashcode()
	n2.Hashcode()
	n3.Hashcode()
	t.Logf("%x, %x, %x, %x", n0.hash, n1.hash, n2.hash, n3.hash)

	set := Nodes{n0, n1, n2, n3, n0, n0}

	set = set.Set()
	correct := Nodes{n0, n1, n2, n3}
	for _, n := range correct {
		assert.Contains(set, n)
	}
	assert.Equal(len(correct), len(set))

	t.Log("Testing intersection")

	set = Nodes{n0, n2, n1, n3} // out of order, on purpose
	other := Nodes{n0, n1}
	inter := set.Intersect(other)

	correct = Nodes{n0, n1}
	for _, n := range correct {
		assert.Contains(inter, n, "inter: %v", inter)
	}
	assert.Equal(len(correct), len(inter))

	t.Log("Testing difference")
	n4 := newNodeFromPool(withGraph(g))
	n5 := newNodeFromPool(withGraph(g))
	set = Nodes{n3, n0, n1, n2}
	other = Nodes{n0, n3, n4, n5}

	diff := set.Difference(other)
	correct = Nodes{n1, n2}
	for _, n := range correct {
		assert.Contains(diff, n)
	}
	assert.Equal(len(correct), len(diff))
}
