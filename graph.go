package gorgonia

import (
	"bytes"
	"fmt"

	"github.com/awalterschulze/gographviz"
	"gonum.org/v1/gonum/graph"
	"gonum.org/v1/gonum/graph/iterator"
)

// ExprGraph is a data structure for a directed acyclic graph (of expressions). This structure is the main entry point
// for Gorgonia.
type ExprGraph struct {
	name string

	all Nodes

	byID   map[int64]int
	byHash map[uint32]*Node
	evac   map[uint32]Nodes
	to     map[*Node]Nodes

	leaves    Nodes
	constants Nodes
	roots     Nodes
	counter   uint
}

// graphconopt sets options
type graphconopt func(g *ExprGraph)

// WithGraphName is a ExprGraph construction option that provides a name.
func WithGraphName(name string) graphconopt {
	f := func(g *ExprGraph) {
		g.name = name
	}
	return f
}

// NewGraph creates a new graph. Duh
func NewGraph(opts ...graphconopt) *ExprGraph {
	g := &ExprGraph{
		byID:   make(map[int64]int),
		byHash: make(map[uint32]*Node),
		evac:   make(map[uint32]Nodes),
		to:     make(map[*Node]Nodes),

		leaves:    make(Nodes, 0, 64),
		constants: make(Nodes, 0, 8),
	}

	for _, opt := range opts {
		opt(g)
	}

	return g
}

// Clone clones the graph. All nodes gets cloned, and their values are cloned as well.
func (g *ExprGraph) Clone() interface{} {
	g2 := new(ExprGraph)
	g2.name = g.name

	mapping := make(map[*Node]*Node) // a map of old nodes to new nodes
	g2.all = make(Nodes, len(g.all))
	for i, n := range g.all {
		cloned := n.Clone().(*Node)
		cloned.g = g2
		cloned.id = n.id

		g2.all[i] = cloned
		mapping[n] = cloned
	}

	// handle each node's children, deriv ofs, etc
	for i, n := range g.all {
		cloned := g2.all[i]
		cloned.children = make(Nodes, len(n.children))
		for j, c := range n.children {
			cloned.children[j] = mapping[c]
		}

		cloned.derivOf = make(Nodes, len(n.derivOf))
		for j, c := range n.derivOf {
			cloned.derivOf[j] = mapping[c]
		}

		if n.deriv != nil {
			cloned.deriv = mapping[n.deriv]
		}
	}

	g2.byID = make(map[int64]int)
	g2.byHash = make(map[uint32]*Node)
	for k, v := range g.byHash {
		g2.byHash[k] = mapping[v]
	}

	g2.evac = make(map[uint32]Nodes)
	for k, v := range g.evac {
		g2.evac[k] = make(Nodes, len(v))
		for i, n := range v {
			g2.evac[k][i] = mapping[n]
		}
	}

	g2.to = make(map[*Node]Nodes)
	for k, v := range g.to {
		to := mapping[k]
		g2.to[to] = make(Nodes, len(v))
		for i, n := range v {
			g2.to[to][i] = mapping[n]
		}
	}

	g2.leaves = make(Nodes, len(g.leaves))
	for i, n := range g.leaves {
		g2.leaves[i] = mapping[n]
	}

	g2.constants = make(Nodes, len(g.constants))
	for i, n := range g.constants {
		g2.constants[i] = mapping[n]
	}

	g2.roots = make(Nodes, len(g.roots))
	for i, n := range g.roots {
		g2.roots[i] = mapping[n]
	}

	g2.counter = g.counter
	return g2
}

// AddNode adds n to the graph. It panics if the added node ID matches an existing node ID.
func (g *ExprGraph) AddNode(n *Node) (retVal *Node) {
	defer func() {
		if _, ok := g.to[retVal]; !ok {
			g.to[retVal] = nil
		}
	}()
	// check for node with the same name in the graph
	// we don't update the graph if this is the case
	for _, node := range g.constants {
		if node.name == n.name && n.isConstant() {
			return node
		}
	}
	hash := n.Hashcode()
	if existing, ok := g.byHash[hash]; ok {
		if existing == nil {
			// this means that there has been previous collisions
			// so look at evac map
			for _, e := range g.evac[hash] {
				if nodeEq(n, e) {
					return e
				}
			}
			g.evac[hash] = append(g.evac[hash], n)
			g.addToAll(n)
			incrCC() // collision counter
			return n
		}

		if !nodeEq(n, existing) {
			g.evac[hash] = Nodes{existing, n}
			g.byHash[hash] = nil // to signal that it's collided
			g.addToAll(n)
			incrCC()
			return n
		}
		incrEC() // expected collision (they're the same node!)
		return existing
	}

	if n.isConstant() {
		n = n.clone()
		g.constants = g.constants.Add(n)
		n.g = g
	}

	g.addToAll(n)
	g.byHash[hash] = n
	return n
}

func (g *ExprGraph) addToAll(n *Node) {
	if n == nil {
		panic("HELP! trying to add nil")
	}
	g.all = append(g.all, n)
	n.id = int64(g.counter)
	g.counter++
}

// RemoveNode removes n from the graph, as well as any edges attached to it. If the node
// is not in the graph it is a no-op.
func (g *ExprGraph) RemoveNode(node graph.Node) {
	n := node.(*Node)
	if n.id == -1 {
		return // if it's -1, it was never in the graph to begin with
	}

	hash := n.Hashcode()

	delete(g.byHash, hash)
	delete(g.to, n)
	g.evac[hash] = g.evac[hash].remove(n)
	g.all = g.all.remove(n)
}

// SetEdge adds e, an edge from one node to another. If the nodes do not exist, they are added.
// It will panic if the IDs of the e.From and e.To are equal.
func (g *ExprGraph) SetEdge(e graph.Edge) {
	from := e.From().(*Node)
	to := e.To().(*Node)

	if from == to {
		panic(fmt.Sprintf("cannot add self edge: from %v to %v", from, to))
	}

	if !g.Has(from.ID()) {
		from = g.AddNode(from)
	}

	if !g.Has(to.ID()) {
		to = g.AddNode(to)
	}

	// g.to[to] = g.to[to].Add(from)
	g.to[to] = append(g.to[to], from)
}

// Roots returns a list of nodes that are not children of any other nodes
func (g *ExprGraph) Roots() (retVal Nodes) {
	// handle subgraph
	if g.roots != nil {
		return g.roots
	}

	for n, tos := range g.to {
		if len(tos) == 0 {
			retVal = append(retVal, n)
		}
		// if the root is a statement (typically a read), and it only has one child
		if len(n.children) == 1 && n.isStmt {
			child := n.children[0]
			if len(g.to[child]) == 1 {
				retVal = append(retVal, child)
			}
		}
	}
	g.roots = retVal
	return retVal
}

// Inputs returns a list of nodes which are inputs (that is to say, the user is required to set a value in it)
func (g *ExprGraph) Inputs() (retVal Nodes) {
	for _, n := range g.all {
		if n.isInput() {
			retVal = append(retVal, n)
		}
	}
	return
}

// UnbindAll unbinds all the values from the nodes
func (g *ExprGraph) UnbindAll() {
	for _, n := range g.all {
		n.unbind()
	}
}

// UnbindAllNonInputs unbinds all the values from nodes that aren't input nodes
func (g *ExprGraph) UnbindAllNonInputs() {
	for _, n := range g.all {
		if n.isInput() || n.isConstant() {
			continue
		}
		n.unbind()
	}
}

// ByName returns nodes that have the name provided.
// Bear in mind that the name that is compared to is the internal name,
// not the result of calling node.Name(). The reason for doing this is
// for ease of finding only names that are user-supplied, instead of autogenerated names
func (g *ExprGraph) ByName(name string) (retVal Nodes) {
	for _, n := range g.all {
		if n.name == name {
			retVal = append(retVal, n)
		}
	}
	return
}

// Constant returns a constant that may be found in the graph. If no constant were found, a new one is created instead
func (g *ExprGraph) Constant(v Value) *Node {
	for _, n := range g.constants {
		if ValueEq(n.Value(), v) {
			return n
		}
	}

	n := NewConstant(v)
	return g.AddNode(n)
}

func (g *ExprGraph) String() string {
	var buf bytes.Buffer
	buf.WriteString("Graph: [\n")
	for _, n := range g.byHash {
		fmt.Fprintf(&buf, "\t%d: %s\n", n.Hashcode(), n)
	}
	buf.WriteString("]")
	return buf.String()
}

// ToDot generates the graph in graphviz format. The use of this is to generate for the entire graph
// which may have multiple trees with different roots
// TODO: This is getting unwieldy. Perhaps refactor out into a ToDot(...Opt)?
func (g *ExprGraph) ToDot() string {
	gv := gographviz.NewEscape()
	gv.SetName(fullGraphName)
	gv.SetDir(true)

	gv.AddAttr(fullGraphName, "nodesep", "1")
	gv.AddAttr(fullGraphName, "ranksep", "1.5 equally")
	gv.AddAttr(fullGraphName, "rankdir", "TB")
	if len(g.byHash) > 100 {
		gv.AddAttr(fullGraphName, "nslimit", "3") // numiter=3*len(nodes)
		// gv.AddAttr(fullGraphName, "splines", "line") // ugly as sin.
	}

	groups := make(map[string]struct{})
	for h, n := range g.byHash {
		if n != nil {
			group := n.dotCluster()
			groups[group] = struct{}{}
			continue
		}
		// other wise it'se a clash of hash
		for _, n := range g.evac[h] {
			group := n.dotCluster()
			groups[group] = struct{}{}

		}
	}

	for grp := range groups {
		attrs := map[string]string{"label": grp}

		parentGraph := fullGraphName
		if grp == inputsClust || grp == constantsClust {
			parentGraph = inputConsts
			if !gv.IsSubGraph(inputConsts) {
				groupAttrs := map[string]string{"rank": "max"}
				gv.AddSubGraph(fullGraphName, inputConsts, groupAttrs)
			}
		}
		gv.AddSubGraph(parentGraph, "cluster_"+grp, attrs)
	}

	// for _, n := range g.byHash {
	for _, n := range g.all {
		group := n.dotCluster()
		n.dotString(gv, "cluster_"+group)
	}

	// for _, from := range g.byHash {
	for _, from := range g.all {
		for i, child := range from.children {
			if ok := g.all.Contains(child); !ok {
				// not in graph, so ignore it...
				continue
			}
			fromID := fmt.Sprintf("Node_%p", from)
			toID := fmt.Sprintf("Node_%p", child)

			edgeAttrs := map[string]string{
				"taillabel":  fmt.Sprintf(" %d ", i),
				"labelfloat": "false",
			}

			// we invert the from and to nodes for gradients, As the expressionGraph builds upwards from bottom, the gradient builds downwards.
			if from.group == gradClust && child.group == gradClust {
				edgeAttrs["dir"] = "back"
				gv.AddPortEdge(toID, toID+":anchor:s", fromID, fromID+":anchor:n", true, edgeAttrs)
			} else {
				gv.AddPortEdge(fromID, fromID+":anchor:s", toID, toID+":anchor:n", true, edgeAttrs)
			}
		}
	}

	// draw deriv lines
	if debugDerives {
		edgeAttrs := map[string]string{
			"style":      "dashed",
			"constraint": "false",
			"weight":     "999",
		}

		for _, n := range g.byHash {
			if n == nil {
				// collision found... what to do?
				continue
			}
			if n.derivOf != nil {
				id := fmt.Sprintf("Node_%p", n)
				for _, derivOf := range n.derivOf {
					if _, ok := g.to[derivOf]; !ok {
						continue
					}
					ofID := fmt.Sprintf("Node_%p", derivOf)
					// gv.AddPortEdge(id, ":anchor:w", ofID, ofID+":anchor:e", true, edgeAttrs)
					gv.AddEdge(id, ofID, true, edgeAttrs)
				}
			}
		}
	}

	// stupid invisible nodes to keep expressiongraph on the left
	subGAttrs := make(map[string]string)
	// subGAttrs.Add("rank", "max")
	gv.AddSubGraph(fullGraphName, outsideSubG, subGAttrs)

	attrs := map[string]string{
		"style": "invis",
	}
	gv.AddNode(outsideSubG, outsideRoot, attrs)

	outsides := []string{outsideRoot}
	var insides []string

	// build the inside and outside list
	if _, hasInputs := groups[inputsClust]; hasInputs {
		insides = append(insides, insideInputs)
		gv.AddNode("cluster_inputs", insideInputs, attrs)
	}

	if _, hasConst := groups[constantsClust]; hasConst {
		if len(insides) > 0 {
			outsides = append(outsides, outsideConsts)
			gv.AddNode(outsideSubG, outsideConsts, attrs)
		}
		insides = append(insides, insideConsts)
		gv.AddNode("cluster_constants", insideConsts, attrs)
	}

	if len(insides) > 0 {
		outsides = append(outsides, outsideExprG)
		gv.AddNode(outsideSubG, outsideExprG, attrs)
	}
	insides = append(insides, insideExprG)
	gv.AddNode("cluster_expressionGraph", insideExprG, attrs)

	for group := range groups {
		if group == exprgraphClust || group == constantsClust || group == inputsClust {
			continue
		}
		inside := "inside_" + group
		outside := "outside_" + group
		insides = append(insides, inside)
		outsides = append(outsides, outside)

		gv.AddNode(outsideSubG, outside, attrs)
		gv.AddNode("cluster_"+group, inside, attrs)
	}

	edgeAttrs := map[string]string{
		"style":      "invis",
		"weight":     "999",
		"constraint": "false",
	}
	for i, o := range outsides {
		// outside-inside
		gv.AddEdge(o, insides[i], true, edgeAttrs)

		if i > 0 {
			// outside-outside
			gv.AddEdge(outsides[i-1], o, true, edgeAttrs)

			// inside-inside
			gv.AddEdge(insides[i-1], insides[i], true, edgeAttrs)
		}
	}
	return gv.String()
}

// Edges returns all the edges in the graph.
func (g *ExprGraph) Edges() graph.Edges {
	var edges []graph.Edge
	for _, n := range g.all {
		for _, toN := range g.to[n] {
			edges = append(edges, edge{
				from: n,
				to:   toN,
			})
		}
	}
	if len(edges) == 0 {
		return graph.Empty
	}
	return iterator.NewOrderedEdges(edges)
}

// other private methods
func (g *ExprGraph) removeAllEdgesFrom(n *Node) {
	for k, ns := range g.to {
		g.to[k] = ns.remove(n)
	}
}

/* Graph interface */

// Node returns the node in the graph with the given ID.
func (g *ExprGraph) Node(id int64) graph.Node {
	// n := (*Node)(unsafe.Pointer(uintptr(id)))
	// for _, n := range g.all {
	// 	if n.id == id {
	// 		return n
	// 	}
	// }
	// return nil
	return g.node(id)
}

func (g *ExprGraph) node(id int64) *Node {
	if idx, ok := g.byID[id]; ok {
		if idx >= len(g.all) {
			return nil
		}
		return g.all[idx]
	}
	for i, n := range g.all {
		if n.id == id {
			g.byID[id] = i
			return n
		}
	}
	return nil
}

// Has returns whether the node exists within the graph.
func (g *ExprGraph) Has(nodeid int64) bool {
	n := g.node(nodeid)
	return n != nil
}

// Nodes returns all the nodes in the graph.
func (g *ExprGraph) Nodes() graph.Nodes {
	// nodes := make([]graph.Node, len(g.from))
	ns := g.AllNodes()

	return nodeToGraphNode(ns)
}

// AllNodes is like Nodes, but returns Nodes instead of []graph.Node.
// Nodes() has been reserved for the graph.Directed interface, so this one is named AllNodes instead
func (g *ExprGraph) AllNodes() Nodes { return g.all }

// From returns all nodes in g that can be reached directly from n.
func (g *ExprGraph) From(nodeid int64) graph.Nodes {
	if n := g.node(nodeid); n != nil {
		return nodeToGraphNode(n.children)
	}
	return nil
}

// HasEdgeBetween returns whether an edge exists between nodes x and y without
// considering direction.
func (g *ExprGraph) HasEdgeBetween(x, y int64) bool {
	xid := g.node(x)
	yid := g.node(y)
	if xid == nil || yid == nil {
		return false
	}

	return xid.children.Contains(yid) || yid.children.Contains(xid)
}

// Edge returns the edge from u to v if such an edge exists and nil otherwise.
// The node v must be directly reachable from u as defined by the From method.
func (g *ExprGraph) Edge(u, v int64) graph.Edge {
	uid := g.node(u)
	vid := g.node(v)

	if uid == nil || vid == nil {
		return nil
	}

	if !uid.children.Contains(vid) {
		return nil
	}
	e := edge{from: uid, to: vid}
	return e
}

/* Directed interface */

// HasEdgeFromTo returns whether an edge exists in the graph from u to v.
func (g *ExprGraph) HasEdgeFromTo(u, v int64) bool {
	uid := g.node(u)
	vid := g.node(v)
	if uid == nil || vid == nil {
		return false
	}

	return uid.children.Contains(vid)
}

// To returns all nodes in g that can reach directly to n.
func (g *ExprGraph) To(nid int64) graph.Nodes {
	n := g.node(nid)
	if n == nil {
		return nil
	}

	ns := g.to[n]
	ns = ns.Set()
	g.to[n] = ns
	return nodeToGraphNode(ns)
}

// subgraph is basically a subset of nodes. This is useful for compiling sub sections of the graph
func (g *ExprGraph) subgraph(ns Nodes, findMissing bool, opts ...Nodes) *ExprGraph {
	// ns = ns.Set()

	var roots Nodes
	// add missing stuff first
	if findMissing {
		for _, n := range ns {
			for _, parent := range g.to[n] {
				if parent.isStmt {
					roots = append(roots, parent)
					ns = append(ns, parent)
				}
			}
		}
	}

	// uniquify the froms and at the same time build a new roots
	allset := ns.mapSet()
	if len(opts) == 0 {
		for _, n := range ns {
			if len(g.to[n]) == 0 {
				if n.isStmt {
					roots = append(roots, n.children[0])
				} else {
					roots = append(roots, n)
				}
				continue
			}

			var hasParent bool
			for _, parent := range g.to[n] {
				if allset.Contains(parent) {
					hasParent = true
					break
				}
			}
			if !hasParent {
				roots = append(roots, n)
			}
		}
	} else {
		rs := opts[0]
		roots = make(Nodes, len(rs))
		for i, n := range rs {
			if n.isStmt {
				roots[i] = n.children[0]
				continue
			}
			roots[i] = n

		}
	}
	var leaves Nodes
	for _, n := range ns {
		if len(n.children) == 0 {
			leaves = append(leaves, n)
		}
	}

	// uniquify all the things
	roots = roots.Set()
	leaves = leaves.Set()
	ns = ns.Set()

	retVal := &ExprGraph{
		all:    ns,
		byID:   make(map[int64]int),
		byHash: g.byHash,
		evac:   g.evac,
		to:     g.to,

		leaves:    leaves,
		constants: g.constants,
		roots:     roots,
	}

	return retVal
}

// Subgraph subsets a graph. This function has overloaded meanings - If only one node is passed in, it assumes that the one node is the root,
// otherwise, it treats ns as the subset of nodes to be included in the subgraph
func (g *ExprGraph) Subgraph(ns ...*Node) *ExprGraph {
	if len(ns) == 1 {
		g.SubgraphRoots(ns[0])
	}
	return g.subgraph(ns, true)
}

// SubgraphRoots creates a subgraph, assuming the provided nodes are roots to the new subgraph.
func (g *ExprGraph) SubgraphRoots(ns ...*Node) *ExprGraph {
	sub := g.walkFromRoots(ns...)
	return g.subgraph(sub, true, ns)
}

// ExactSubgraphRoots creates a subgraph from the roots provided.
// The difference between SubgraphRoots and ExactSubgraphRoots is that ExactSubGraphRoots
// will not attempt to discover if any nodes are missing.
//
// Given a function like the following:
//		z = x + y
//		set(x, -x.Grad) // setting the value of x to the negative of the gradient
//
// When SubgraphRoots is used on z, the `-x.Grad` will be included.
// When using ExactSubgraphRoots, only `x` and `y` are included in the subgraph
func (g *ExprGraph) ExactSubgraphRoots(ns ...*Node) *ExprGraph {
	sub := g.walkFromRoots(ns...)
	return g.subgraph(sub, false, ns)
}

func (g *ExprGraph) walkFromRoots(ns ...*Node) Nodes {
	sub := make(Nodes, len(ns))
	copy(sub, ns)

	walked := NewNodeSet()
	for _, n := range ns {
		ch := make(chan *Node)
		go func(ch chan *Node) {
			defer close(ch)
			walkGraph(n, ch, walked)
		}(ch)

		for node := range ch {
			sub = append(sub, node)
		}
	}
	return sub
}

type edge struct {
	from, to graph.Node
	weight   float64
}

func (e edge) From() graph.Node         { return e.from }
func (e edge) To() graph.Node           { return e.to }
func (e edge) ReversedEdge() graph.Edge { e.from, e.to = e.to, e.from; return e }
func (e edge) Weight() float64          { return e.weight }
