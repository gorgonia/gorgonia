package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// node is compatible with graph.Node and onnx.DataCarrier
type node struct {
	id        int64
	t         tensor.Tensor
	operation *onnx.Operation
	name      string
	// gorgoniaNode stores a pointer to the node of the exprgraph
	gorgoniaNode *gorgonia.Node
}

// ID to fulfill the graph.Node interface
func (n *node) ID() int64 {
	return n.id
}

// SetTensor assign the tensor N to the underlying node
func (n *node) SetTensor(t tensor.Tensor) error {
	n.t = t
	if n.gorgoniaNode != nil {
		err := gorgonia.Let(n.gorgoniaNode, t)
		if err != nil {
			return err
		}
	}
	return nil
}

// GetTensor value from the node
func (n *node) GetTensor() tensor.Tensor {
	return n.t
}
