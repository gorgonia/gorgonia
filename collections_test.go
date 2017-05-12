package gorgonia

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNodes(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	n0 := newNode(In(g), WithName("n0"))
	n1 := newNode(In(g), WithName("n1"))
	n2 := newNode(In(g), WithName("n2"))
	n3 := newNode(In(g), WithName("n3"))

	// calculate hashcode first
	n0h := n0.Hashcode()
	n1h := n1.Hashcode()
	n2h := n2.Hashcode()
	n3h := n3.Hashcode()
	t.Logf("%x, %x, %x, %x", n0.hash, n1.hash, n2.hash, n3.hash)

	set := Nodes{n0, n1, n2, n3, n0, n0}

	set = set.Set()
	correct := Nodes{n0, n1, n2, n3}
	for _, n := range correct {
		assert.Contains(set, n, "SET: %v", set)
	}
	assert.Equal(len(correct), len(set))

	t.Log("Test add")
	set = Nodes{}
	set = set.Add(n0)
	set = set.Add(n2)
	set = set.Add(n0)
	set = set.Add(n3)
	set = set.Add(n1)
	correct = Nodes{n0, n2, n3, n1}
	assert.Equal(correct, set)

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
	n4 := newNode(In(g))
	n5 := newNode(In(g))
	set = Nodes{n3, n0, n1, n2}
	other = Nodes{n0, n3, n4, n5}

	diff := set.Difference(other)
	correct = Nodes{n1, n2}
	for _, n := range correct {
		assert.Contains(diff, n)
	}
	assert.Equal(len(correct), len(diff))

	t.Log("Testing replace")
	set = Nodes{n0, n2, n1, n2, n1} // not yet a set
	set = set.replace(n2, n3)
	correct = Nodes{n0, n3, n1, n3, n1}
	assert.Equal(correct, set)

	t.Log("Formatting")
	formats := []string{"% v", "%+v", "%d", "%v", "%#v", "%Y", "%P"}
	correctFormats := []string{
		"[n0  n1  n2  n3]",
		`[n0, 
n1, 
n2, 
n3]`,
		fmt.Sprintf("[%x, %x, %x, %x]", n0h, n1h, n2h, n3h),
		"[n0, n1, n2, n3]",
		"[n0 :: <nil>, n1 :: <nil>, n2 :: <nil>, n3 :: <nil>]",
		"[<nil>, <nil>, <nil>, <nil>]",
		fmt.Sprintf("[%p, %p, %p, %p]", n0, n1, n2, n3),
	}

	set = Nodes{n0, n1, n2, n3}
	for i, f := range formats {
		s := fmt.Sprintf(f, set)
		if s != correctFormats[i] {
			t.Errorf("Format %q. Expected %q. Got %q", f, correctFormats[i], s)
		}
	}

	// corner cases
	set = Nodes{}
	if set.AllSameGraph() {
		t.Error("Empty list of nodes cannot be of the same graph!")
	}

	nAbnormal := newNode(In(NewGraph()))
	set = Nodes{n0, n1, nAbnormal, n2}
	if set.AllSameGraph() {
		t.Error("One node is in a different graph! This should have returned false")
	}
}
