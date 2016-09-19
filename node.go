package gorgonia

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"hash"
	"hash/fnv"
	"unsafe"

	"github.com/awalterschulze/gographviz"
	tf32 "github.com/chewxy/gorgonia/tensor/f32"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"
	"github.com/chewxy/gorgonia/tensor/types"
)

// A Node is a node in the computation graph
type Node struct {
	// metadata of the node
	t     Type // pruned types only plz
	shape types.Shape

	// this node is the result of applying the op to the children
	op       Op
	children Nodes // shortcut, instead of having to go through the graph

	// For nicely grouping stuff in graphviz.
	// TODO: Should this be in *Node?
	name  string
	group string

	g *ExprGraph // this node belongs in this graph

	// value bondage
	// inputs are bound to values directly
	boundTo Value

	// to track derivations
	derivOf Nodes
	deriv   *Node

	// for hashing nodes
	hash uint32

	hashed        bool
	inferredShape bool // is shape inferred?
	unchanged     bool // has this node been modified
	isStmt        bool // is this a statment node
	ofInterest    bool // is this node of particular interest? (for debugging)
}

// NodeConsOpt is a function that provides construction options for any Node
type NodeConsOpt func(*Node)

func withType(t Type) NodeConsOpt {
	f := func(n *Node) {
		n.t = t
	}
	return f
}

func withChildren(children Nodes) NodeConsOpt {
	f := func(n *Node) {
		n.children = children
	}
	return f
}

func withOp(op Op) NodeConsOpt {
	f := func(n *Node) {
		n.op = op
		if _, ok := op.(stmtOp); ok {
			n.isStmt = true
		}
	}
	return f
}

func withGraph(g *ExprGraph) NodeConsOpt {
	f := func(n *Node) {
		n.g = g
	}
	return f
}

// WithName is a node creation option that gives the *Node the provided name. This is especially useful in debugging graphs.
func WithName(name string) NodeConsOpt {
	f := func(n *Node) {
		n.name = name
	}
	return f
}

// WithValue is a node creation option that binds the value to the *Node.
func WithValue(any interface{}) NodeConsOpt {
	v, err := anyToValue(any)
	if err != nil {
		panic(err)
	}

	f := func(n *Node) {
		if !typeEq(v.Type(), n.t) {
			panic(fmt.Sprintf("TypeError: Want %v, Got %v instead", n.t, v.Type())) // yes this is a runtime error
		}
		n.bind(v)
		if n.shape == nil {
			n.shape = v.Shape()
		}
	}
	return f
}

// WithInit is a node creation option to initialize a *Node with the InitWFn provided.
func WithInit(fn InitWFn) NodeConsOpt {
	f := func(n *Node) {
		dt, err := dtypeOf(n.t)
		if err != nil {
			panic(err)
		}

		var T types.Tensor
		var v Value
		switch dt {
		case Float64:
			val := fn(dt, n.shape...).([]float64)
			T = tf64.NewTensor(tf64.WithShape(n.shape...), tf64.WithBacking(val))
			v = FromTensor(T)
		case Float32:
			val := fn(dt, n.shape...).([]float32)
			T = tf32.NewTensor(tf32.WithShape(n.shape...), tf32.WithBacking(val))
			v = FromTensor(T)
		default:
			panic("Not handled yet")
		}
		WithValue(v)(n)
	}
	return f
}

// WithShape is a node creation option to initialize a *Node with a particular shape
func WithShape(shp ...int) NodeConsOpt {
	s := types.Shape(shp)
	f := func(n *Node) {
		if n.Dims() != s.Dims() {
			panic(fmt.Sprintf("Node %v is a %v, which has %d dimensions. Input shape of %p is %v, which has %d dimensions", n, n.t, n.Dims(), n, s, s.Dims()))
		}
		n.shape = s
	}
	return f
}

// WithGroupName is a node creation option to group a *Node within a particular group. This option is useful for debugging with graphs
func WithGroupName(name string) NodeConsOpt {
	f := func(n *Node) {
		if n.group == "" {
			n.group = name
		}
	}
	return f
}

// the function is here because there are some init() calls that requires it
func newNode(opts ...NodeConsOpt) *Node {
	n := new(Node)
	for _, opt := range opts {
		opt(n)
	}
	n.fix()
	n.fixChildren()
	n.fixEdges()

	incrNN()
	return n
}

func newNodeFromPool(opts ...NodeConsOpt) *Node {
	n := borrowNode()
	for _, opt := range opts {
		opt(n)
	}
	n.fix()

	incrNN() // number of new nodes requested
	return n
}

func newUniqueNode(opts ...NodeConsOpt) *Node {
	n := newNodeFromPool(opts...)
	if n.g == nil {
		return n
	}
	n.fixChildren() // ensure that all the kids are in the graph first

	m := n.g.AddNode(n)
	if n != m {
		returnNode(n)
	}
	m.fixEdges()
	return m
}

// ID returns the ID of the node. This satisfies the gonum/graph.Node interface
func (n *Node) ID() int { return int(uintptr(unsafe.Pointer(n))) }

// helper functions to help compilation process
func (n *Node) isArg() bool      { return n.op == nil }
func (n *Node) isInput() bool    { return n.isArg() && !n.isStmt }
func (n *Node) isMutable() bool  { return !n.isInput() && n.op.returnsPtr() }
func (n *Node) isConstant() bool { _, ok := n.op.(constant); return ok }

func (n *Node) isRoot() bool {
	if n.g == nil {
		return true
	}
	return len(n.g.to[n]) == 0
}

// type related isX() helper methods

// IsScalar indicates if a node represents a a scalar value. This is based on the type of the node, not the actual value associated with the node
func (n *Node) IsScalar() bool { _, ok := n.t.(Dtype); return ok }

// IsVector indicates if a node represents a vector value. This is based on the type of the node, not the actual value associated with the node
func (n *Node) IsVector() bool {
	if t, ok := n.t.(*TensorType); ok {
		return t.d == 1
	}

	return false
}

// IsColVec indicates if a node represents a Column Vector. This is based on the type of the node, not the actual value associated with the node
func (n *Node) IsColVec() bool {
	if _, ok := n.t.(*TensorType); ok {
		if n.shape != nil {
			return n.shape.IsColVec()
		}
	}
	return false
}

// IsRowVec indicates if a node represents a Row Vector. This is based on the type of the node, not the actual value associated with the node
func (n *Node) IsRowVec() bool {
	if _, ok := n.t.(*TensorType); ok {
		if n.shape != nil {
			return n.shape.IsRowVec()
		}
	}
	return false
}

// IsMatrix indicates if a node represents a matrix. This is based on the type of the node, not the actual value associated with the node
func (n *Node) IsMatrix() bool {
	if t, ok := n.t.(*TensorType); ok {
		return t.d == 2
	}
	return false
}

// methods

// CloneTo clones the node into a new graph. If CloneTo() is called on the same graph as the n, it will return n. The reason this is done is because
// at any given time, every node  should be unique in the *ExprGraph.
//
//TODO: clone children as well (this means that CloneTo() is only currently suitable fo input nodes)
func (n *Node) CloneTo(g *ExprGraph) *Node {
	if n.g != nil && g == n.g {
		return n
	}

	n2 := newNodeFromPool(withGraph(g), withOp(n.op), WithName(n.name), withType(n.t))
	if n.shape != nil {
		n2.shape = n.shape.Clone()
		n2.inferredShape = n.inferredShape
	}

	if n.boundTo != nil {
		var err error
		if n2.boundTo, err = n.boundTo.clone(); err != nil {
			panic(err)
		}
	}
	n2 = g.AddNode(n2)
	return n2
}

// Value returns the valuse bound to the node. May return nil
func (n *Node) Value() Value {
	if n.isConstant() {
		return n.op.(constant).Value()
	}
	if dv, ok := n.boundTo.(*dualValue); ok {
		return dv.Value
	}
	return n.boundTo
}

// Grad returns the gradient if there is one.
func (n *Node) Grad() (Value, error) {
	if dv, ok := n.boundTo.(*dualValue); ok {
		return dv.d, nil
	}
	if n.deriv != nil {
		return n.deriv.Value(), nil
	}

	return nil, NewError(GraphError, "No Gradient node/value found for %v", n)
}

// Dims indicates how many dimensions the node's result has
func (n *Node) Dims() int { return n.t.dims() }

// Shape returns the shape of the node
func (n *Node) Shape() types.Shape { return n.shape }

// IsVec returns whether this node is a vector
func (n *Node) IsVec() bool { return n.IsVector() }

// Name returns the name of the node. If a name was specified and it is too long,
// the short name will be used instead (except in inputs)
//
// The short name is typically of the form: OpName(%1, %2 ...), making it read more like a function call
func (n *Node) Name() string {
	if n.name != "" {
		return n.name
	}

	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%s(", n.op)
	for i, child := range n.children {
		fmt.Fprintf(&buf, "%%%x", child.Hashcode())
		if i < len(n.children)-1 {
			buf.WriteString(", ")
		}
	}
	buf.WriteString(")")
	return buf.String()
}

func (n *Node) WriteHash(h hash.Hash32) {
	fmt.Fprintf(h, "%v%v", n.t, n.shape)

	if n.isInput() {
		h.Write([]byte(n.name))
	} else {

		n.op.WriteHash(h)
	}

	// if len(n.children) == 0 {
	// 	binary.Write(h, binary.LittleEndian, byte(0))
	// }

	binary.Write(h, binary.LittleEndian, byte(len(n.children)))
	for _, child := range n.children {
		binary.Write(h, binary.LittleEndian, child.Hashcode())
	}

}

// Hashcode() provides the hash for the tree, assuming that the node is the root of the tree.
// Original implementation was here by Vatine (who's apparently 80 years old and using SO!?!):
//		http://stackoverflow.com/questions/1988665/hashing-a-tree-structure
func (n *Node) Hashcode() uint32 {
	if n.hashed {
		return n.hash
	}
	h := fnv.New32a()
	n.WriteHash(h)
	n.hash = h.Sum32()
	n.hashed = true
	return n.hash
}

// ToDot returns the graph as a graphviz compatible string
func (n *Node) ToDot() string {
	graphName := exprgraphClust

	g := gographviz.NewEscape()
	g.SetName(graphName)
	g.SetDir(true)

	g.AddAttr(exprgraphClust, "splines", "spline")
	g.AddAttr(exprgraphClust, "nodesep", "0.5")
	g.AddAttr(exprgraphClust, "ranksep", "1.2 equally")

	seen := make(map[*Node]string)
	n.dot(g, graphName, seen)

	return g.String()
}

// RestrictedToDot prints the graphviz compatible string but does not print the entire tree
// up and down indicates how many levels to look up, and how many levels to look down
func (n *Node) RestrictedToDot(up, down int) string {
	if n.g == nil {
		return n.ToDot()
	}

	g := n.g
	var ns, upQ, downQ Nodes

	//	up
	ns = Nodes{n}
	upQ = Nodes{n}
	for l := 0; l < up; l++ {
		origLen := len(upQ)
		for i := 0; i < origLen; i++ {
			qn := upQ[i]
			toQN := graphNodeToNode(g.To(qn))
			upQ = append(upQ, toQN...)
			ns = append(ns, toQN...)
		}
		upQ = upQ[origLen:]
	}

	// down
	downQ = Nodes{n}
	for d := 0; d < down; d++ {
		origLen := len(downQ)
		for i := 0; i < origLen; i++ {
			qn := downQ[i]
			downQ = append(downQ, qn.children...)
			ns = append(ns, qn.children...)
		}
		downQ = downQ[origLen:]
	}

	sg := g.subgraph(ns)

	n.ofInterest = true
	defer func() {
		n.ofInterest = false
	}()
	return sg.ToDot()
}

// String() implements the fmt.Stringer interface
func (n *Node) String() string {
	var buf bytes.Buffer
	if n.Name() != "" {
		fmt.Fprintf(&buf, "%s :: ", n.Name())
	} else {
		fmt.Fprintf(&buf, "%s :: ", n.op)
	}
	if c, ok := n.op.(constant); ok {
		fmt.Fprintf(&buf, "%v{%v}", n.t, c.Value())
	} else {
		fmt.Fprintf(&buf, "%v", n.t)
	}
	return buf.String()
}

// private methods

// TODO: check type, check shape, check if needsGrad -> promote to dualValue
func (n *Node) bind(v Value) error {
	if n.boundTo == nil {
		n.boundTo = v
		return nil
	}

	if dv, ok := n.boundTo.(*dualValue); ok {
		if vdv, ok := v.(*dualValue); ok {
			if vdv == dv {
				return nil
			}
			panic("Undefined behaviour") // no seriously there literally is no defined behaviour of what should the right thing be. I'll come back to this TODO.
		}
		dv.Value = v
		return nil
	}

	n.boundTo = v

	return nil
}

// unbind releases the values back to the pool
func (n *Node) unbind() {
	if n.boundTo == nil {
		return
	}

	if dv, ok := n.boundTo.(*dualValue); ok {
		returnDV(dv)
	}

	if t, ok := n.boundTo.(Tensor); ok {
		returnTensor(t)
	}
	n.boundTo = nil
}

func (n *Node) dotCluster() string {
	var group string
	var isConst bool
	var isInput = n.isInput()

	if n.op != nil {
		_, isConst = n.op.(constant)
	}

	switch {
	case isConst:
		group = constantsClust
	case isInput:
		group = inputsClust
	case n.group == "":
		group = exprgraphClust
	default:
		group = n.group
	}
	return group
}

func (n *Node) dot(g *gographviz.Escape, graphName string, seen map[*Node]string) string {
	var id string
	var ok bool
	if id, ok = seen[n]; !ok {
		id = n.dotString(g, graphName)
		seen[n] = id
	} else {
		return id
	}

	for i, child := range n.children {
		childID := child.dot(g, graphName, seen)
		edgeAttrs := gographviz.NewAttrs()
		edgeAttrs.Add("taillabel", fmt.Sprintf(" %d ", i+1))
		edgeAttrs.Add("labelfloat", "false")
		// edgeAttrs.Add("dir", "back")

		g.AddPortEdge(id, id+":anchor:s", childID, childID+":anchor:n", true, edgeAttrs)
	}
	return id
}

func (n *Node) fix() {
	if n.IsScalar() {
		n.shape = scalarShape
	}

	if n.isConstant() {
		return
	}

	if n.g == nil {
		panic(fmt.Sprintf("no graph supplied %v", n))
	}
}

func (n *Node) fixChildren() {
	if n.g == nil {
		return
	}

	for i, child := range n.children {
		newChild := n.g.AddNode(child)
		if child != newChild {
			n.children[i] = newChild
		}
	}
}

func (n *Node) fixEdges() {
	if n.g == nil {
		return
	}

	if len(n.children) > 0 {
		for _, child := range n.children {
			e := edge{from: n, to: child}
			n.g.SetEdge(e)
		}
	} else {
		n.g.leaves = append(n.g.leaves, n)
	}
}

func (n *Node) setShape(s types.Shape, inferred bool) {
	n.shape = s
	n.inferredShape = inferred
}

func (n *Node) setGroup(grp string) {
	n.group = grp
}

func (n *Node) clone(opts ...NodeConsOpt) *Node {
	if n.isInput() {
		return n
	}

	nn := newNodeFromPool(withChildren(n.children),
		withType(n.t),
		withOp(n.op),
		WithName(n.name),
		withGraph(n.g),
	)

	for _, opt := range opts {
		opt(nn)
	}

	// if the shape is already known...
	if n.shape != nil {
		nn.shape = n.shape
		nn.inferredShape = n.inferredShape
	}

	return nn
}

func (n *Node) diffWRT() []bool {
	if n.op == nil {
		return nil
	}

	return n.op.DiffWRT(len(n.children))
}

// dfs but does not use channels. useful for extracting paths. used particularly in test
func (n *Node) seqWalk() Nodes {
	retVal := Nodes{n}
	for _, child := range n.children {
		retVal = append(retVal, child.seqWalk()...)
	}
	return retVal
}

// dotString returns the ID of the node.
func (n *Node) dotString(g *gographviz.Escape, graphName string) string {
	var buf bytes.Buffer
	if err := exprNodeTempl.ExecuteTemplate(&buf, "node", n); err != nil {
		panic(err)
	}

	id := fmt.Sprintf("Node_%p", n)
	label := buf.String()
	attrs := gographviz.NewAttrs()
	attrs.Add("fontname", "monospace")
	attrs.Add("shape", "none")
	attrs.Add("label", label)

	g.AddNode(graphName, id, attrs)
	return id
}
