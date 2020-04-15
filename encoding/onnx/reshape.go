package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type reshape struct{}

func init() {
	register("Reshape", newReshape)
}

func newReshape() operator {
	return &reshape{}
}

func (a *reshape) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 2)
	if err != nil {
		return err
	}

	var toShape tensor.Shape
	if to, ok := children[1].gorgoniaNode.Value().Data().([]int64); ok {
		toShape = make([]int, len(to))
		for i := 0; i < len(to); i++ {
			toShape[i] = int(to[i])
		}
	} else {
		return fmt.Errorf("Cannot reshape, bad output shape %#v", children[1].gorgoniaNode.Value().Data())
	}
	n.gorgoniaNode, err = gorgonia.Reshape(children[0].gorgoniaNode, toShape)

	return err
}

func (a *reshape) init(o onnx.Operation) error {
	return nil
}
