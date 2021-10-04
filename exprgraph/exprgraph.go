package exprgraph

import (
	"fmt"
	"sort"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/exprgraph/internal/uid"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
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
	MinNodeID = 1
)

// Graph is a representation of an expression graph.
// The graph is a directed weighted graph.
// The lighter the link, the lefter the operand.
// For example a graph to represent c = a (op) b
// is represented as:
//  digraph {
//      c -> a [weight 0]
//      c -> b [weight 1]
//  }
type Graph struct {
	tensor.Engine
	nodes  map[int64]*Node
	from   map[int64]map[int64]graph.WeightedEdge
	to     map[int64]map[int64]graph.WeightedEdge
	groups map[int64]encoding.Groups

	self, absent float64

	nodeIDs uid.Set
}

// Graph returns itself
func (g *Graph) Graph() *Graph { return g }

// NameOf returns the name of the given Tensor.
// It returns an error if the tensor is not found in the graph
func (g *Graph) NameOf(t Tensor) (string, error) {
	n := g.find(t)
	if n == nil {
		return "", errors.Wrapf(ErrNotFoundInGraph, "Cannot find name of %v", t)
	}
	return n.name, nil
}

// IDOf returns the ID of the given Tensor.
// it returns -1 if the node is not found
func (g *Graph) IDOf(t Tensor) (NodeID, error) {
	n := g.find(t)
	if n == nil {
		return -1, errors.Wrapf(ErrNotFoundInGraph, "Cannot find ID of %v", t)
	}
	return NodeID(n.id), nil
}

// NodeOf returns the node of the given Tensor.
// it returns nil if the node is not found
func (g *Graph) NodeOf(t Tensor) *Node { return g.find(t) }

func (g *Graph) find(t Tensor) *Node {
	// search backwards because it's more probable that you're using newer created nodes
	for _, n := range g.nodes {
		// this little trick here (to inspect the internal structure - i.e g.nodes[i].Tensor == t)
		// is the real reason why you cannot really create Node{Node{Node{...}}}
		// without doing it explicitly
		if tt, ok := n.Tensor.(Tensor); ok {
			if t == tt {
				return n
			}
		}
		if n.beforeLift != nil {
			if tt, ok := n.beforeLift.(Tensor); ok {
				if t == tt {
					return n
				}
			}
		}
		if t == n {
			return n
		}
	}
	return nil
}

// NewGraph with default values
func NewGraph(e tensor.Engine) *Graph {
	if e == nil {
		e = tensor.StdEng{}
	}
	return &Graph{
		Engine: e,
		nodes:  make(map[int64]*Node),
		from:   make(map[int64]map[int64]graph.WeightedEdge),
		to:     make(map[int64]map[int64]graph.WeightedEdge),
		groups: make(map[int64]encoding.Groups),

		self:   Self,
		absent: Absent,

		nodeIDs: uid.NewSet(),
	}
}

// Node returns the node with the given ID, if it exists. Nil otherwise.
func (g *Graph) Node(id int64) graph.Node {
	if n, ok := g.nodes[id]; ok {
		return n
	}
	return nil
}

// Nodes returns the list of all nodes in the graph.
func (g *Graph) Nodes() graph.Nodes {
	if len(g.nodes) == 0 {
		return graph.Empty
	}
	ordered := make([]*Node, len(g.nodes))
	for id, n := range g.nodes {
		ordered[id] = n
	}
	return NodesFromOrdered(ordered)
}

// From returns the list of nodes that can be reached directly from the given ID.
func (g *Graph) From(id int64) graph.Nodes {
	if len(g.from[id]) == 0 {
		return graph.Empty
	}
	edges := g.from[id]
	es := make(byWeight, 0, len(edges))
	for _, e := range edges {
		es = append(es, e)
	}
	sort.Sort(es)
	ns := make([]*Node, 0, len(es))
	for _, e := range es {
		ns = append(ns, g.nodes[e.To().ID()])
	}
	return NodesFromOrdered(ns)

}

// HasEdgeBetween returns whether an edge exists between x and y.
func (g *Graph) HasEdgeBetween(xid, yid int64) bool {
	if _, ok := g.from[xid][yid]; ok {
		return true
	}
	_, ok := g.from[yid][xid]
	return ok
}

// Edge returns an edge object, if an edge exists. Nil otherwise.
func (g *Graph) Edge(uid, vid int64) graph.Edge {
	return g.WeightedEdge(uid, vid)
}

// WeightedEdge returns the weighted edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *Graph) WeightedEdge(uid, vid int64) graph.WeightedEdge {
	edge, ok := g.from[uid][vid]
	if !ok {
		return nil
	}
	return edge
}

// HasEdgeFromTo returns whether a directed edge between x and y.
func (g *Graph) HasEdgeFromTo(uid, vid int64) bool {
	if _, ok := g.from[uid][vid]; !ok {
		return false
	}
	return true
}

// To returns all the nodes that can reach the given id.
func (g *Graph) To(id int64) graph.Nodes {
	if len(g.to[id]) == 0 {
		return graph.Empty
	}

	edges := g.to[id]
	es := make(byWeight, 0, len(edges))
	for _, e := range edges {
		es = append(es, e)
	}
	sort.Sort(es)
	ns := make([]*Node, 0, len(es))
	for _, e := range es {
		ns = append(ns, g.nodes[e.From().ID()])
	}
	return NodesFromOrdered(ns)
}

// NewNode returns a new Node with a unique
// arbitrary ID and default values.
func (g *Graph) NewNode() *Node {
	if len(g.nodes) == 0 {
		return &Node{
			id: MinNodeID,
		}
	}
	if int64(len(g.nodes)) == uid.Max {
		panic("simple: cannot allocate node: no slot")
	}
	return &Node{
		id: g.nodeIDs.NewID(),
	}
}

// AddNode adds a node to the graph. AddNode panics if
// the added node ID matches an existing node ID.
func (g *Graph) AddNode(n *Node) error {
	if n.ID() < MinNodeID {
		return errors.New("Cannot add a node with an ID less than MinNodeID")
	}
	if _, exists := g.nodes[n.ID()]; exists {
		return fmt.Errorf("simple: node ID collision: %d", n.ID())
	}
	if l, ok := n.Tensor.Engine().(Lifter); ok {
		t := n.Tensor
		n.beforeLift = t
		n.Tensor = l.Lift(t)
	}
	g.nodes[n.ID()] = n
	g.nodeIDs.Use(n.ID())
	return nil
}

// setWeightedEdge adds a weighted edge from one node to another. If the nodes do not exist, they are added
// and are set to the nodes of the edge otherwise.
// It will return an error if the IDs of the e.From and e.To are equal.
// It will return an  ErrNotFoundInGraph is either from or to node does not exists in the graph
func (g *Graph) setWeightedEdge(e graph.WeightedEdge) error {
	var (
		from = e.From()
		fid  = from.ID()
		to   = e.To()
		tid  = to.ID()
	)

	if fid == tid {
		return errors.New("simple: adding self edge")
	}

	if _, ok := g.nodes[fid]; !ok {
		return fmt.Errorf("node id %v: %w", fid, ErrNotFoundInGraph)
	}
	if _, ok := g.nodes[tid]; !ok {
		return fmt.Errorf("node id %v: %w", tid, ErrNotFoundInGraph)
	}
	g.nodes[fid] = from.(*Node)
	g.nodes[tid] = to.(*Node)

	if fm, ok := g.from[fid]; ok {
		fm[tid] = e
	} else {
		g.from[fid] = map[int64]graph.WeightedEdge{tid: e}
	}
	if tm, ok := g.to[tid]; ok {
		tm[fid] = e
	} else {
		g.to[tid] = map[int64]graph.WeightedEdge{fid: e}
	}
	return nil
}

// AddChildren creates weighted edges betwen n and children.
// The function returns an error if a child is not present in the graph
// or if any link betwen n and one of children already exists
func (g *Graph) AddChildren(n *Node, children ...*Node) error {
	if _, ok := g.nodes[n.ID()]; !ok {
		return fmt.Errorf("%q: %w", n, ErrNotFoundInGraph)
	}
	for _, n := range children {
		if _, ok := g.nodes[n.ID()]; !ok {
			return fmt.Errorf("%v: %w", n, ErrNotFoundInGraph)
		}
	}
	for i, child := range children {
		err := g.setWeightedEdge(WeightedEdge{
			F: n,
			T: child,
			W: float64(i), // the weight of the child is the order of the children into the Op
		},
		)
		if err != nil {
			return err
		}
	}
	return nil
}

// Roots returns the roots of a graph.
func (g *Graph) Roots() (retVal Nodes) {
	for n, tos := range g.to {
		if len(tos) == 0 {
			retVal.ns = append(retVal.ns, g.nodes[n])
		}
	}
	return retVal
}

// SetGroup sets the group of the given node.
func (g *Graph) SetGroup(n *Node, group encoding.Group) { g.groups[n.id].Upsert(group) }
