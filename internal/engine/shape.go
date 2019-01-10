package engine

import "gorgonia.org/tensor"

var scalarShape = tensor.ScalarShape()

type axes []int
type coordinates []int

// only works for 2D
func transpose2D(shape tensor.Shape) tensor.Shape {
	if len(shape) != 2 {
		return shape
	}
	retVal := tensor.BorrowInts(2)
	retVal[0] = shape[1]
	retVal[1] = shape[0]
	return retVal
}

// for batched matmul
func transposeBatch2D(shape tensor.Shape) tensor.Shape {
	if len(shape) != 3 {
		return shape
	}
	retVal := tensor.BorrowInts(3)
	retVal[0] = shape[0]
	retVal[1] = shape[2]
	retVal[2] = shape[1]
	return retVal
}
