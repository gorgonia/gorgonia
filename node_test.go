package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNodeBasics(t *testing.T) {
	var n *Node
	var err error
	var c0, c1 *Node
	g := NewGraph()

	// withGraph
	n, err = newNode(In(g))
	if err != nil {
		t.Fatal(err)
	}
	if n.g == nil {
		t.Error("Expected *Node to be constructed with a graph")
	}
	returnNode(n)

	// withType
	n, err = newNode(In(g), WithType(Float64))
	if err != nil {
		t.Fatal(err)
	}
	if !n.t.Eq(Float64) {
		t.Error("Expected *Node to be constructed with Float64")
	}
	returnNode(n)

	// withOp
	n, err = newNode(In(g), WithOp(newEBOByType(addOpType, Float64, Float64)))
	if err != nil {
		t.Fatal(err)
	}
	if op, ok := n.op.(elemBinOp); ok {
		if op.binOpType() != addOpType {
			t.Error("expected addOpType")
		}
	} else {
		t.Error("Expected *Node to be constructed with an addOp")
	}
	returnNode(n)

	// withOp - statement op
	n, err = newNode(In(g), WithOp(letOp{}))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := n.op.(letOp); ok {
		if !n.isStmt {
			t.Errorf("Expected *Node.isStmt to be true when a statement op is passed in")
		}
	} else {
		t.Error("Expected  *Node to be constructed with a letOp")
	}
	returnNode(n)

	// WithName
	n, err = newNode(In(g), WithName("n"))
	if err != nil {
		t.Fatal(err)
	}
	if n.name != "n" {
		t.Error("Expected *Node to be constructed with a name \"n\"")
	}
	returnNode(n)

	// withChildren
	c0, _ = newNode(In(g), WithName("C0"))
	c1, _ = newNode(In(g), WithName("C1"))
	n, _ = newNode(In(g), WithChildren(Nodes{c0, c1}))
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
	n, err = newNode(In(g), WithChildren(Nodes{c0}))
	if err != nil {
		t.Fatal(err)
	}
	if len(n.children) != 1 {
		t.Error("Expected *Node to have 1 child")
	}
	returnNode(n)
	returnNode(c0)

	n, err = newNode(In(g), WithValue(F64(3.14)), WithGrad(F64(1)))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := n.boundTo.(*dualValue); !ok {
		t.Error("Expected a dual Value")
	}
	returnNode(n)

	// WithValue but no type
	n, err = newNode(In(g), WithValue(F64(3.14)))
	if err != nil {
		t.Fatal(err)
	}
	if !n.t.Eq(Float64) {
		t.Error("Expected a *Node to be constructed WithValue to get its type from the value if none exists")
	}
	if !ValueEq(n.boundTo, newF64(3.14)) {
		t.Error("Expected *Node to be bound to the correct value. Something has gone really wrong here")
	}
	returnNode(n)

	// WithValue but with existing type that is the same
	n, err = newNode(In(g), WithType(Float64), WithValue(F64(3.14)))
	if err != nil {
		t.Fatal(err)
	}
	if !ValueEq(n.boundTo, newF64(3.14)) {
		t.Error("Expected *Node to be bound to the correct value. Something has gone really wrong here")
	}
	returnNode(n)

	// This is acceptable and should not panic
	n, err = newNode(In(g), WithType(makeTensorType(1, Float64)), WithShape(2, 1))
	if err != nil {
		t.Fatal(err)
	}
	returnNode(n)

	// Returns itsef
	n, err = newNode(In(g), WithType(makeTensorType(2, Float32)), WithShape(2, 3))
	if err != nil {
		t.Fatal(err)
	}
	m := n.Node()
	if n != m {
		t.Error("Expected n.Node() to return itself, pointers and all")
	}
	ns := n.Nodes()
	if len(ns) != 1 {
		t.Errorf("Expected Nodes() to return a slice of length 1. Got %v", ns)
	}
	if ns[0] != n {
		t.Error("Expected first slice to be itself.")
	}
	m = nil
	returnNode(n)

	// bad stuff

	// no graph
	n, err = newNode(WithType(Float64))
	if err == nil {
		t.Fail()
	}
	n, err = newNode(In(g), WithType(Float32), WithValue(F64(1)))
	if err == nil {
		t.Fail()
	}

	// type mismatch - values were set first
	n, err = newNode(In(g), WithValue(F64(1)), WithType(Float32))
	if err == nil {
		t.Fail()
	}

	// shape type mismatch
	n, err = newNode(In(g), WithType(makeTensorType(1, Float64)), WithShape(2, 2))
	if err == nil {
		t.Fail()
	}

	// bad grads
	n, err = newNode(WithGrad(F64(3.14)))
	if err == nil {
		t.Fail()
	}
}

func TestNewUniqueNodes(t *testing.T) {
	var n *Node
	var c0, c1 *Node
	g := NewGraph()

	// withChildren but they're constants
	c0 = NewConstant(3.14)
	var err error
	c1, err = newNode(In(g), WithValue(5.0))
	if err != nil {
		t.Fatal(err)
	}
	n = NewUniqueNode(In(g), WithChildren(Nodes{c0, c1}))
	if n.children[0].g == nil {
		t.Error("Expected a cloned constant child to have graph g")
	}

	returnNode(n)
}

func TestCloneTo(t *testing.T) {
	g := NewGraph()
	g2 := NewGraph()

	n := NewUniqueNode(WithName("n"), WithType(Float64), In(g))
	n.CloneTo(g2)

	assert.True(t, nodeEq(g2.AllNodes()[0], n))
}

func TestWithShape(t *testing.T) {
	/*
		vecShapeFunc := WithShape([]int{2})
		colVecShapeFunc := WithShape([]int{2, 1})
		rowVecShapeFunc := WithShape([]int{1, 2})
		matrixShapeFunc := WithShape([]int{2, 2})
		singleMatrixShapeFunc := WithShape([]int{1, 1})
	*/
}
