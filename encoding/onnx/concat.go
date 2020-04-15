package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

type concat struct {
	axis int
}

func init() {
	register("Concat", newConcat)
}

func newConcat() operator {
	return &concat{}
}

func (a *concat) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	var nodes = make([]*gorgonia.Node, len(children))
	for i := 0; i < len(children); i++ {
		nodes[i] = children[i].gorgoniaNode
	}
	var err error
	n.gorgoniaNode, err = gorgonia.Concat(a.axis, nodes...)
	return err
}

func (a *concat) init(o onnx.Operation) error {
	axis, ok := o.Attributes["axis"]
	if !ok {
		return errors.New("concat: expected axis attribute is not found")
	}
	err := errors.New("axis in not an int")
	if axis, ok := axis.(int64); ok {
		a.axis = int(axis)
		err = nil
	}
	return err
}
