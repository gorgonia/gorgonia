package onnx

import (
	"errors"
	"log"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
)

// Reshape a tensor
type Reshape struct{}

// Constructor to fulfil the interface ...
func (r *Reshape) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
		it := getOrderedChildren(g, n)
		// Get the shape from the child
		if it.Len() != 2 {
			return nil, errors.New("invalid number of children, expected 2")
		}
		children := make([]*engine.Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*engine.Node)
		}
		shape := children[1]
		log.Println(shape.Value().Data())
		var s []int
		for _, v := range shape.Value().Data().([]int64) {
			s = append(s, int(v))
		}

		return engine.NewReshapeOperation(s)(g, n.(*engine.Node))
	}
}
