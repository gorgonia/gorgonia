package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/tensor"
)

// Graph is a representation of an expression graph
type Graph struct {
	// A Graph implements a StandardEngine.
	tensor.StdEng

	// adj is an adjacency list
	adj [][]NodeID

	// nodes is a flat list of nodes
	nodes []Node

	// track derivatives

	// for each node indexed by the id, which other node is a derivative?
	// e.g if
	// 	deriv[1] = 3
	// that means that node ID 1 has node ID 3 as a deriv.
	deriv []NodeID

	// for each node indexed by the id, of which other nodes is it a derivative of
	// e.g. if
	// 	derivOf[3] = []ID{1,2}
	// that means that node ID 3 is the deriv of nodes with ID 1 and 2.
	derivOf [][]NodeID

	// flags
	isStdEng bool
}

// Node returns the node with the given ID, if it exists. Nil otherwise.
func (g *Graph) Node(id int64) graph.Node {
	if int(id) >= len(g.nodes) || id < 0 {
		return nil
	}
	return g.nodes[int(id)]
}

// Nodes returns the list of all nodes in the graph.
func (g *Graph) Nodes() graph.Nodes { return &iterator{nodes: g.nodes} } // TODO: copy g.nodes?

// From returns the list of nodes that can be reached directly from the given ID.
func (g *Graph) From(id int64) Nodes { panic("NYI") }

// HasEdgeBetween returns whether an edge exists between x and y.
func (g *Graph) HasEdgeBetween(x, y int64) bool { panic("NYI") }

// Edge returns an edge object, if an edge exists. Nil otherwise.
func (g *Graph) Edge(x, y int64) graph.Edge { panic("NYI") }

// HasEdgeFromTo returns whether a directed edge between x and y.
func (g *Graph) HaEdgeFromTo(x, y int64) bool { panic("NYI") }

// To returns all the nodes that can reach the given id.
func (g *Graph) To(id int64) graph.Nodes { panic("NYI") }

/* unexported methods */

// id returns the ID of the given tensor.
func (g *Graph) id(t tensor.Tensor) int64 {
	// search backwards because it's more probable that you're using newer created nodes
	for i := len(g.nodes) - 1; i >= 0; i-- {
		if t == g.nodes[i].Tensor {
			return int64(i)
		}
	}
	return -1
}

// idOrInsert returns the ID of the given tensor if the tensor is in the expression graph. Otherwise it adds it.
func (g *Graph) idOrInsert(t tensor.Tensor) int64 {
	id := g.id(t)
	if id < 0 {
		return g.insert(t)
	}
	return id
}

// insert inserts the tensor into the expression graph, and returns the ID
func (g *Graph) insert(t tensor.Tensor) int64 {
	l := len(g.nodes)
	g.nodes = append(g.nodes, Node{Tensor: t})
	g.nodes[l].NodeID = NodeID(l)
	return int64(l)
}

// nameOf returns the name of the tensor.
func (g *Graph) nameOf(t tensor.Tensor) string {
	id := g.id(t)
	// TODO: if not found?
	return g.nodes[id].name
}

// name gives a name to a tensor in the expression graph
func (g *Graph) name(t tensor.Tensor, s string) error {
	id := g.id(t)
	g.nodes[id].name = s

	return nil //TODO: if not found
}

// nodeOf returns the node of the given tensor. If not found _______ TODO
func (g *Graph) nodeOf(t tensor.Tensor) Node {
	id := g.id(t)
	return g.nodes[id] // TODO: if not found?
}

func (g *Graph) addChildren(id int64, children []int64) { panic("NYI") }
