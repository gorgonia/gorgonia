package gorgonnx

import (
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// populateExprgraph by walking through the graph
func (g *graph) populateExprgraph() error {
	// Walk the graph
	itN := g.Nodes()
	nodes := make([]*Node, 0, itN.Len())
	for itN.Next() {
		// if the node is a "tensor", set it!
		n := itN.Node().(*Node)
		if n.t != nil && n.gorgoniaNode == nil && n.operation == nil {
			n.gorgoniaNode = gorgonia.NodeFromAny(g.exprgraph, n.t, gorgonia.WithName(uuid.New().String()))
		} else {
			nodes = append(nodes, n)
		}
	}
	for len(nodes) > 0 {
		initialLen := len(nodes)
		for i := 0; i < len(nodes); i++ {
			n := nodes[i]
			if n.operation != nil {
				children := getOrderedChildren(g.g, n)
				nilChild := false
				for j := 0; j < len(children); j++ {
					if children[j].gorgoniaNode == nil {
						nilChild = true
						break
					}
				}
				if nilChild {
					continue
				}
				err := g.applyOperation(n)
				if err != nil {
					return err
				}
				nodes = append(nodes[:i], nodes[i+1:]...)
			}
		}
		if len(nodes) == initialLen {
			return errors.New("infinite loop")
		}
	}
	return nil
}

// applyOperation creates a new node on the exprgraph
func (g *graph) applyOperation(n *Node) error {
	// Is this node already in the ExprGraph?
	if n.gorgoniaNode != nil {
		return fmt.Errorf("unsupported case: node is already in the exprgraph")
	}
	var op operator
	var opC func() operator
	var ok bool
	if opC, ok = operators[n.operation.Name]; !ok {
		return &onnx.ErrNotImplemented{
			Operator: n.operation.Name,
		}
	}
	op = opC()
	err := op.init(*n.operation)
	if err != nil {
		return err
	}
	return op.apply(g, n)
}
