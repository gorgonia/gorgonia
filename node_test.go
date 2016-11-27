package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNodeBasics(t *testing.T) {
	var n *Node
	var c0, c1 *Node
	g := NewGraph()

	// withGraph
	n = newNode(withGraph(g))
	if n.g == nil {
		t.Error("Expected *Node to be constructed with a graph")
	}
	returnNode(n)

	// withType
	n = newNode(withGraph(g), withType(Float64))
	if !n.t.Eq(Float64) {
		t.Error("Expected *Node to be constructed with Float64")
	}
	returnNode(n)

	// withOp
	n = newNode(withGraph(g), withOp(newEBOByType(addOpType, Float64, Float64)))
	if op, ok := n.op.(elemBinOp); ok {
		if op.binOpType() != addOpType {
			t.Error("expected addOpType")
		}
	} else {
		t.Error("Expected *Node to be constructed with an addOp")
	}
	returnNode(n)

	// withOp - statement op
	n = newNode(withGraph(g), withOp(letOp{}))
	if _, ok := n.op.(letOp); ok {
		if !n.isStmt {
			t.Errorf("Expected *Node.isStmt to be true when a statement op is passed in")
		}
	} else {
		t.Error("Expected  *Node to be constructed with a letOp")
	}
	returnNode(n)

	// WithName
	n = newNode(withGraph(g), WithName("n"))
	if n.name != "n" {
		t.Error("Expected *Node to be constructed with a name \"n\"")
	}
	returnNode(n)

	// withChildren
	c0 = newNode(withGraph(g), WithName("C0"))
	c1 = newNode(withGraph(g), WithName("C1"))
	n = newNode(withGraph(g), withChildren(Nodes{c0, c1}))
	if len(n.children) == 2 {
		if !n.children.Contains(c0) || !n.children.Contains(c1) {
			t.Error("Expected *Node to contain those two children")
		}
	} else {
		t.Error("Expected *Node to be constructed with 2 children")
	}
	if !n.isRoot() {
		t.Error("n is supposed to be root")
	}

	returnNode(n)
	returnNode(c0)
	returnNode(c1)

	// withChildren but they're constants
	c0 = NewConstant(3.14)
	n = newNode(withGraph(g), withChildren(Nodes{c0}))
	if len(n.children) != 1 {
		t.Error("Expected *Node to have 1 child")
	}
	returnNode(n)
	returnNode(c0)

	// WithValue but no type
	n = newNode(withGraph(g), WithValue(F64(3.14)))
	if !n.t.Eq(Float64) {
		t.Error("Expected a *Node to be constructed WithValue to get its type from the value if none exists")
	}
	if !ValueEq(n.boundTo, F64(3.14)) {
		t.Error("Expected *Node to be bound to the correct value. Something has gone really wrong here")
	}
	returnNode(n)

	// WithValue but with existing type that is the same
	n = newNode(withGraph(g), withType(Float64), WithValue(F64(3.14)))
	if !ValueEq(n.boundTo, F64(3.14)) {
		t.Error("Expected *Node to be bound to the correct value. Something has gone really wrong here")
	}
	returnNode(n)

	// bad stuff
	var f func()

	// no graph
	f = func() {
		n = newNode(withType(Float64))
	}
	assert.Panics(t, f)

	// conflicting types, types were set first
	f = func() {
		n = newNode(withGraph(g), withType(Float32), WithValue(F64(1)))
	}
	assert.Panics(t, f)

	// type mismatch - values were set first
	f = func() {
		n = newNode(withGraph(g), WithValue(F64(1)), withType(Float32))
	}
	assert.Panics(t, f)

	// shape type mismatch
	f = func() {
		n = newNode(withGraph(g), withType(newTensorType(1, Float64)), WithShape(2, 1))
	}
	assert.Panics(t, f)
}

func TestNewUniqueNodes(t *testing.T) {
	var n *Node
	var c0, c1 *Node
	g := NewGraph()

	// withChildren but they're constants
	c0 = NewConstant(3.14)
	c1 = newNode(withGraph(g), WithValue(5.0))
	n = newUniqueNode(withGraph(g), withChildren(Nodes{c0, c1}))
	if n.children[0].g == nil {
		t.Error("Expected a cloned constant child to have graph g")
	}

	returnNode(n)

}
