package gorgonia

import (
	"errors"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/distro"
	"gorgonia.org/gorgonia/internal/engine"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/tensor"
)

// START_GRAPH OMIT

// Graph ...
type Graph struct {
	g *engine.ExprGraph
}

// NewGraph ...
func NewGraph() *Graph {
	return &Graph{g: engine.NewGraph()}
}

// END_GRAPH OMIT

// NewScalar creates a Node representing a variable that holds a scalar value
func NewScalar(g *Graph, t tensor.Dtype, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NewScalar(t, opts...)
	g.g.AddNode(n)
	return n
}

// NodeFromAny creates a Node from a tensor.Tensor, automatically filling in shape and type info
func NodeFromAny(g *Graph, any interface{}, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NodeFromAny(any, opts...)
	g.g.AddNode(n)
	return n
}

// NewVector creates a Node representing a variable that holds a vector (nx1 matrix)
func NewVector(g *Graph, t tensor.Dtype, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NewVector(t, opts...)
	g.g.AddNode(n)
	return n
}

// NewMatrix creates a Node representing a variable that holds a matrix (nxm)
func NewMatrix(g *Graph, t tensor.Dtype, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NewMatrix(t, opts...)
	g.g.AddNode(n)
	return n
}

// NewTensor creates a Node representing a variable that holds a tensor (any n-dimensional array with dimensions greater than 2)
func NewTensor(g *Graph, t tensor.Dtype, dims int, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NewTensor(t, dims, opts...)
	g.g.AddNode(n)
	return n
}

// NewConstant takes in any reasonable value and makes it a constant node.
func NewConstant(g *Graph, v interface{}, opts ...engine.NodeConsOpt) node.Node {
	n := g.g.NewConstant(v, opts...)
	g.g.AddNode(n)
	return n
}

// Let binds a value.Value to a node that is a variable. A variable is represented as a *Node with no Op.
// It is equivalent to :
//		x = 2
func Let(n node.Node, be interface{}) error {
	_, ok := n.(*engine.Node)
	if !ok {
		return errors.New("Note an engine node")
	}
	return engine.Let(n.(*engine.Node), be)
}

// WithType is a node construction option to set a node to the specified type.
// Types in *Node are immutable once set. If the type has already been specified in the node,
// a check will be made to see if the both types are the same. If it isn't, it will panic.
func WithType(t hm.Type) engine.NodeConsOpt {
	return engine.WithType(t)
}

// WithName is a node construction option that gives the *Node the provided name. This is especially useful in debugging graphs.
func WithName(name string) engine.NodeConsOpt {
	return engine.WithName(name)
}

// WithValue is a node construction option that binds the value to the *Node. This function may panic if:
//	- Gorgonia was unable to convert interface{} into a Value.
//	- The type of the Value does not match the type of the nodes.
func WithValue(any interface{}) engine.NodeConsOpt {
	return engine.WithValue(any)
}

// WithGrad is a node construction option that binds the value to the *Node. This function may panic if:
//	- There isn't already a value associated with the node (.boundTo == nil)
//	- The type of the Value does not match the value of the node.
func WithGrad(any interface{}) engine.NodeConsOpt {
	return engine.WithGrad(any)
}

// WithInit is a node construction option to initialize a *Node with the InitWFn provided.
func WithInit(fn distro.InitWFn) engine.NodeConsOpt {
	return engine.WithInit(fn)
}

// WithShape is a node construction option to initialize a *Node with a particular shape.
// This function panics if the shape's dimensions do not match the specified dimensions of the *Node.
func WithShape(shp ...int) engine.NodeConsOpt {
	return engine.WithShape(shp...)
}

// WithGroupName is a node construction option to group a *Node within a particular group. This option is useful for debugging with graphs.
func WithGroupName(name string) engine.NodeConsOpt {
	return engine.WithGroupName(name)
}
