package exprgraph

import (
	"fmt"
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/exprgraph/internal/uid"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
	stdeng "gorgonia.org/tensor/engines"
)

// constraints
var (
	_ graph.Directed = &Graph{}
)

const (
	// Self is the weight of a self link in a Graph
	Self float64 = -10
	// Absent is the weight of a missing link in a graph
	Absent float64 = -20
	// MinNodeID is the minimun ID of the node accepted in the graph
	MinNodeID = 0
)

// Graph is a representation of an expression graph.
// For example, if a graph were to represent the following expression:
//
//	c = a (OP) b
//
// then the graph would look like this:
//
//	c → a
//	c → b
//
// Internally, the graph is destructured into several smaller data structures, to aid with
// ease of retrieval.
//
// The `nodes` map is a container holding all the Nodes and mapping them to their ID.
//
// The `from` map maps an ID to a list of IDs. So the example above will look like this:
//
//	from: map[int64][]int64 {
//		c.ID(): []int64{a.ID(), b.ID()}
//	}
//
// The slice index represents the order of things (i.e. `a` comes before `b`)
//
// The `to` map maps an ID to a list of IDs that came from it. So the example above will look like this:
//
//	to: map[int64][]int64{
//		a.ID(): []int64{c.ID()},
//		b.ID(): []int64{c.ID()},
//	}
type Graph struct {
	tensor.Engine
	nodes  map[int64]Node
	from   map[int64][]int64         // from node to list of nodes. The key is the `from`.
	to     map[int64][]int64         // to node from a a list of nodes. The key is the `to`
	groups map[int64]encoding.Groups // a node (int64) is in these groups (encoding.Groups)

	self, absent float64

	nodeIDs uid.Set
}

// Graph returns itself.
func (g *Graph) Graph() *Graph { return g }

// NameOf returns the name of the given Tensor.
// It returns an error if the tensor is not found in the graph
func (g *Graph) NameOf(t Tensor) (string, error) {
	n := g.find(t)
	if n == nil {
		return "", errors.Wrapf(ErrNotFoundInGraph, "Cannot find name of %v", t)
	}
	return n.Name(), nil
}

// IDOf returns the ID of the given Tensor.
// it returns -1 if the node is not found
func (g *Graph) IDOf(t Tensor) (NodeID, error) {
	n := g.find(t)
	if n == nil {
		return -1, errors.Wrapf(ErrNotFoundInGraph, "Cannot find ID of %v", t)
	}
	return NodeID(n.ID()), nil
}

// NodeOf returns the node of the given Tensor.
// it returns nil if the node is not found
func (g *Graph) NodeOf(t Tensor) Node { return g.find(t) }

func (g *Graph) find(t Tensor) Node {
	if t == nil {
		return nil
	}
	if n, ok := t.(Node); ok {
		return n
	}

	// search backwards because it's more probable that you're using newer created nodes
	for _, n := range g.nodes {
		switch nx := n.(type) {
		case valuelifter:
			// this little trick here (to inspect the internal structure - i.e g.nodes[i].Tensor == t)
			// is the real reason why you cannot really create Node{Node{Node{...}}}
			// without doing it explicitly
			tt := nx.Value().(Tensor)
			if t == tt {
				return n
			}
			b4 := nx.prelift()
			if b4 != nil {
				tt := b4.(Tensor)
				if t == tt {
					return n
				}
			}
		default:
			if t == n {
				return n
			}

		}

	}
	return nil
}

func (g *Graph) getByID(id int64) Node {
	if n, ok := g.nodes[id]; ok {
		return n
	}
	return nil
}

// NewGraph with default values
func NewGraph(e tensor.Engine) *Graph {
	if e == nil {
		e = stdeng.Gen{}
	}
	g := &Graph{
		Engine: e,
		nodes:  make(map[int64]Node),
		from:   make(map[int64][]int64),
		to:     make(map[int64][]int64),
		groups: make(map[int64]encoding.Groups),

		self:   Self,
		absent: Absent,

		nodeIDs: uid.NewSet(),
	}
	if s, ok := e.(graphSetter); ok {
		s.SetGraph(g)
	}
	return g
}

// Get gets the concrete Node of a nodelike (graph.Node) object.
func (g *Graph) Get(n Nodelike) Node { return g.getByID(n.ID()) }

// Node returns the node with the given ID, if it exists. Nil otherwise.
func (g *Graph) Node(id int64) graph.Node {
	n := g.getByID(id)
	// This weird bit here is necessary because (Node)(nil) ≠ nil (untyped)
	if n == nil { // (Node)(nil)
		return nil // untyped nil
	}
	return n
}

// Nodes returns the list of all nodes in the graph.
func (g *Graph) Nodes() graph.Nodes {
	if len(g.nodes) == 0 {
		return graph.Empty
	}

	ordered := make([]Node, 0, len(g.nodes))
	for _, n := range g.nodes {
		ordered = append(ordered, n)
	}
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].ID() < ordered[j].ID() })
	return IterNodesFromNodes(ordered)
}

// From returns the list of nodes that can be reached directly from the given ID.
func (g *Graph) From(id int64) graph.Nodes {
	if len(g.from[id]) == 0 {
		return graph.Empty
	}
	return IterNodesFromIDs(g, g.from[id])
}

// HasEdgeBetween returns whether an edge exists between x and y.
func (g *Graph) HasEdgeBetween(xid, yid int64) bool {
	if in(g.from[xid], yid) {
		return true
	}

	if in(g.to[xid], yid) {
		return true
	}
	return false
}

// Edge returns an edge object, if an edge exists. Nil otherwise.
func (g *Graph) Edge(uid, vid int64) graph.Edge { return g.WeightedEdge(uid, vid) }

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *Graph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	for i, v := range g.from[uid] {
		if v == vid {
			n := g.nodes[uid]
			child := g.nodes[vid]
			return &WeightedEdge{
				F: n,
				T: child,
				W: float64(i), // the weight of the child is the order of the children into the Op
			}
		}
	}
	return nil
}

// HasEdgeFromTo returns whether a directed edge between x and y.
func (g *Graph) HasEdgeFromTo(uid, vid int64) bool { return in(g.from[uid], vid) }

// To returns all the nodes that can reach the given id.
func (g *Graph) To(id int64) graph.Nodes {
	if len(g.to[id]) == 0 {
		return graph.Empty
	}
	return IterNodesFromIDs(g, g.to[id])
}

// newNodeID is to be called to get a new ID for nodes
func (g *Graph) newNodeID() NodeID {
	if len(g.nodes) == 0 {
		return MinNodeID
	}
	if int64(len(g.nodes)) == uid.Max {
		panic("simple: cannot allocate node: no slot")
	}
	return NodeID(g.nodeIDs.NewID())
}

// // newNode creates a new Node with a unique
// // arbitrary ID and default values.
// func (g *Graph) newNode() Node {
// 	if len(g.nodes) == 0 {
// 		return &Node{
// 			id: MinNodeID,
// 		}
// 	}
// 	if int64(len(g.nodes)) == uid.Max {
// 		panic("simple: cannot allocate node: no slot")
// 	}
// 	return &Node{
// 		id: g.nodeIDs.NewID(),
// 	}
// }

// AddNode adds a node to the graph. AddNode panics if
// the added node ID matches an existing node ID.
func (g *Graph) AddNode(n Node) error {
	if n.ID() < MinNodeID {
		return errors.New("Cannot add a node with an ID less than MinNodeID")
	}
	if _, exists := g.nodes[n.ID()]; exists {
		return fmt.Errorf("simple: node ID collision: %d", n.ID())
	}
	n = liftNode(n)
	g.nodes[n.ID()] = n
	g.nodeIDs.Use(n.ID())
	return nil
}

// createEdge creates an edge.
func (g *Graph) createEdge(from, to Node) error {
	if from == to || from.ID() == to.ID() {
		return errors.New("Adding self-edge")
	}

	// this check is not necessary because createEdge is only called by AddChildren
	// which checks for existence of the graph nodes alredy
	/*
		fid, tid := from.ID(), to.ID()
		if _, ok := g.nodes[fid]; !ok {
			return fmt.Errorf("node id %v: %w", fid, ErrNotFoundInGraph)
		}
		if _, ok := g.nodes[tid]; !ok {
			return fmt.Errorf("node id %v: %w", tid, ErrNotFoundInGraph)
		}
	*/

	fid := from.ID()
	tid := to.ID()
	// We don't create multiple edges. No hypergraph shennanigans here

	if in(g.from[fid], tid) {
		return nil
	}
	if in(g.to[tid], fid) {
		return nil
	}

	g.from[fid] = append(g.from[fid], tid)
	g.to[tid] = append(g.to[tid], fid)
	return nil

}

// AddChildren creates weighted edges betwen n and children.
// The function returns an error if a child is not present in the graph
// or if any link betwen n and one of children already exists
func (g *Graph) AddChildren(n Node, children ...Node) error {
	if _, ok := g.nodes[n.ID()]; !ok {
		return fmt.Errorf("%q: %w", n, ErrNotFoundInGraph)
	}
	for _, n := range children {
		if _, ok := g.nodes[n.ID()]; !ok {
			return fmt.Errorf("%v: %w", n, ErrNotFoundInGraph)
		}
	}

	for i, child := range children {
		if err := g.createEdge(n, child); err != nil {
			return errors.Wrapf(err, "Adding edge between %v and %v (%dth child)", n, child, i)
		}
	}
	return nil
}

// Roots returns the roots of a graph.
func (g *Graph) Roots() (retVal []Node) {
	for id, n := range g.nodes {
		if len(g.to[id]) == 0 {
			retVal = append(retVal, n)
		}
	}
	return retVal
}

// SetGroup sets the group of the given node.
func (g *Graph) SetGroup(t Tensor, group encoding.Group) {
	n := g.find(t)
	id := n.ID()
	g.groups[id] = g.groups[id].Upsert(group)
}

// GroupsOf returns the groups that a tensor belongs to.
func (g *Graph) GroupsOf(t Tensor) encoding.Groups {
	n := g.find(t)
	return g.groups[n.ID()]
}

/* UTILITY API METHODS FOR PERFORMANCE */
/* These methods are not strictly necessary.

   Consider `.ChildrenOf()`.
   You can use the original API to find get a `[]NodeID`:
   	iterChildren := g.From(n.ID())
        childNodes := iterChildren.NodeIDs()

   Here you will note that `g.From` allocates a *IterNodes (which allocates a []Node).
   Then `.NodeIDs` will allocate a `[]NodeID`.
   If we're only ever going to use a slice of NodeID, then the intermediary allocation
   of `[]Node` (in the *IterNodes) is a wasted allocation.
*/

// edgesOf gets an edge of a Nodelike. You must provide a map of edges (usually g.to or g.from).
func (g *Graph) edgesOf(t Nodelike, whichEdges map[int64][]int64) []int64 {
	if t == nil {
		return nil
	}

	n := g.getByID(t.ID())
	if n == nil {
		return nil
	}

	return whichEdges[n.ID()]
}

// ChildrenOf finds the children of a given tensor. The result is returned as a NodeIDs.
// The returned value is a copy, so it can be safely mutated upon.
func (g *Graph) ChildrenOf(t Nodelike) NodeIDs {
	children := g.edgesOf(t, g.from)
	if len(children) == 0 {
		return nil
	}

	retVal := make(NodeIDs, 0, len(children))
	for _, child := range children {
		retVal = append(retVal, NodeID(child))
	}
	return retVal
}

// ChildrenOfAsNodes finds the children of a given tensor. The result is returned as a NodeIDs.
func (g *Graph) ChildrenOfAsNodes(t Nodelike) []Node {
	children := g.edgesOf(t, g.from)
	if len(children) == 0 {
		return nil
	}

	retVal := make([]Node, 0, len(children))
	for _, child := range children {
		retVal = append(retVal, g.getByID(child))
	}
	return retVal
}

// ParentsOf finds the parents of a given tensor. The result is returned as a NodeIDs.
func (g *Graph) ParentsOf(t Nodelike) NodeIDs {
	parents := g.edgesOf(t, g.to)
	if len(parents) == 0 {
		return nil
	}
	retVal := make(NodeIDs, 0, len(parents))
	for _, p := range parents {
		retVal = append(retVal, NodeID(p))
	}
	return retVal
}

// ParentsOfAsNodes returns the parents of a given tensor as a []Node.
func (g *Graph) ParentsOfAsNodes(t Nodelike) []Node {
	parents := g.edgesOf(t, g.to)
	if len(parents) == 0 {
		return nil
	}

	retVal := make([]Node, 0, len(parents))
	for _, p := range parents {
		retVal = append(retVal, g.getByID(p))
	}
	return retVal
}
