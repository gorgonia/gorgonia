package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

type matMul struct{}

func init() {
	register("MatMul", newMatMul)
}

func newMatMul() operator {
	return &matMul{}
}

func (a *matMul) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}
	if len(children[0].gorgoniaNode.Shape()) > 2 || len(children[1].gorgoniaNode.Shape()) > 2 {
		return &onnx.ErrNotImplemented{
			Operator: "Matmul",
			Message:  "dimension too high",
		}
	}

	n.gorgoniaNode, err = gorgonia.Mul(children[0].gorgoniaNode, children[1].gorgoniaNode)

	return err
}

func (a *matMul) init(o onnx.Operation) error {
	return nil
}
