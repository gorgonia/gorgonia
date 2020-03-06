package exprgraph

import "gonum.org/v1/gonum/graph"

// Graph is a representation of an expression graph
type Graph struct {
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
