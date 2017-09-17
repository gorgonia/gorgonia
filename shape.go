package gorgonia

import "gorgonia.org/tensor"

var scalarShape = tensor.ScalarShape()

type axes []int
type coordinates []int

// only works for 2D
func transpose2D(shape tensor.Shape) tensor.Shape {
	if len(shape) != 2 {
		return shape
	}

	return tensor.Shape{shape[1], shape[0]}
}
