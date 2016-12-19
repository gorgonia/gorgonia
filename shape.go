package gorgonia

import "github.com/chewxy/gorgonia/tensor/types"

var scalarShape = types.ScalarShape()

type axes []int
type coordinates []int

// only works for 2D
func transpose2D(shape types.Shape) types.Shape {
	if len(shape) != 2 {
		return shape
	}

	return types.Shape{shape[1], shape[0]}
}
