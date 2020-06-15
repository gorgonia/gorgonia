package exprgraph

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Graph is a representation of an expression graph
type Graph struct {
	// A Graph implements a StandardEngine.
	tensor.StandardEngine

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
type graphSetter interface {
	SetGraph(g *Graph)
}

// New creates a new *Graph.
func New(eng tensor.StandardEngine) *Graph {
	g := &Graph{}
	if gs, ok := eng.(graphSetter); ok {
		gs.SetGraph(g)
	}
	g.StandardEngine = eng
	g.nodes = make([]Node, 0, 1024)
	return g
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

/* functions dealing with data in the graph */

// Insert inserts a gorgonia.Tensor and returns the Node ID.
func (g *Graph) Insert(t gorgonia.Tensor) NodeID { return NodeID(g.idOrInsert(t)) }

// NameOf returns the name of a given tensor
func (g *Graph) NameOf(t gorgonia.Tensor) string { return g.nameOf(t) }

// Name associates a name with a given gorgonia.
func (g *Graph) Name(t gorgonia.Tensor, s string) { g.name(t, s) }

// ID returns the ID of the given gorgonia.
func (g *Graph) ID(t gorgonia.Tensor) NodeID {
	// search backwards because it's more probable that you're using newer created nodes
	for i := len(g.nodes) - 1; i >= 0; i-- {
		// this little trick here (to inspect the internal structure - i.e g.nodes[i].Tensor == t)
		// is the real reason why you cannot really create Node{Node{Node{...}}}
		// without doing it explicitly
		if t == g.nodes[i] || t == g.nodes[i].Tensor {
			return NodeID(i)
		}
	}
	return -1
}

func (g *Graph) AddChildren(id NodeID, children []NodeID) {
	diff := int(id) - len(g.adj)
	if diff >= 0 {
		g.adj = append(g.adj, make([][]NodeID, diff+1)...)
	}
	g.adj[id] = append(g.adj[id], children...)
}

/* Monoidy stuff */

func (g *Graph) Graph() *Graph { return g }

/* unexported methods */

// idOrInsert returns the ID of the given tensor if the tensor is in the expression graph. Otherwise it adds it.
func (g *Graph) idOrInsert(t gorgonia.Tensor) NodeID {
	id := g.ID(t)
	if id < 0 {
		return g.insert(t)
	}
	return id
}

// insert inserts the tensor into the expression graph, and returns the ID
func (g *Graph) insert(t gorgonia.Tensor) NodeID {
	l := len(g.nodes)
	g.nodes = append(g.nodes, Node{Tensor: t})
	g.nodes[l].NodeID = NodeID(l)
	return g.nodes[l].NodeID
}

// nameOf returns the name of the gorgonia.
func (g *Graph) nameOf(t gorgonia.Tensor) string {
	id := g.ID(t)
	// TODO: if not found?
	return g.nodes[id].name
}

// name gives a name to a tensor in the expression graph
func (g *Graph) name(t gorgonia.Tensor, s string) error {
	id := g.ID(t)
	g.nodes[id].name = s

	return nil //TODO: if not found
}

// nodeOf returns the node of the given gorgonia. If not found _______ TODO
func (g *Graph) nodeOf(t gorgonia.Tensor) Node {
	id := g.ID(t)
	return g.nodes[id] // TODO: if not found?
}
