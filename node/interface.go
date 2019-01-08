package node

import (
	"github.com/chewxy/hm"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// Node is an abstration of a node of the ExprGraph
type Node interface {
	graph.Node
	// Device returns the device the data will be on
	Device() execution.Device
	// Dims indicates how many dimensions the node's result has
	Dims() int
	// Dtype returns the dtype of the node
	Dtype() tensor.Dtype
	// Grad returns the gradient if there is one.
	Grad() (value.Value, error)
	// GradOnDevice gets the gradient value of the node as a value.Value but on the desired device. In this build the device is always CPU, so it's equivalent to calling .Grad()
	GradOnDevice(dev execution.Device, extern execution.External) (retVal value.Value, allocOnExtern bool, err error)
	// IsColVec indicates if a node represents a Column Vector. This is based on the type of the node, not the actual value associated with the node
	IsColVec() bool
	// IsMatrix indicates if a node represents a matrix. This is based on the type of the node, not the actual value associated with the node
	IsMatrix() bool
	// IsRowVec indicates if a node represents a Row Vector. This is based on the type of the node, not the actual value associated with the node
	IsRowVec() bool
	// IsScalar indicates if a node represents a a scalar value. This is based on the type of the node, not the actual value associated with the node
	IsScalar() bool
	// IsVar returns true if the node represents a differentiable variable (i.e. it's an argument to the function that is not a statement)
	IsVar() bool
	// IsVec returns whether this node is a vector
	IsVec() bool
	// IsVector indicates if a node represents a vector value. This is based on the type of the node, not the actual value associated with the node
	IsVector() bool
	// Shape returns the shape of the node
	Shape() tensor.Shape
	// Strides returns the strides of the value of the node
	Strides() []int
	// Type returns the type of the node
	Type() hm.Type
	// Value returns the valuse bound to the node. May return nil
	Value() value.Value
	// ValueOnDevice gets the value of the node as a value.Value but on the desired device. In this build the device is always CPU, so it's equivalent to calling .Value()
	ValueOnDevice(dev execution.Device, extern execution.External) (retVal value.Value, allocOnExtern bool, err error)
}
