package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

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

// KeepDims is a function that ensures that input and output dimensions are the same though the shape may change.
//
// The expandLeft flag in the function indicates if any shape expansion should be done leftwards or rightwards.
// For example, if fn() returns a tensor with a shape (3) and the desired dimension is 2,
// then if `expandLeft` is true the result will be `(1, 3)`. Otherwise the result will be `(3, 1)`.
//
// At the moment, results that turn into scalars cannot have their dimensions kept - the semantics isn't well established yet and is a work in progress.
func KeepDims(a *Node, expandLeft bool, fn func(a *Node) (*Node, error)) (*Node, error) {
	oshape := a.Shape()
	adims := oshape.Dims()
	b, err := fn(a)
	if err != nil {
		return nil, err
	}

	// happy path = quick exit
	newShape := b.Shape()
	if newShape.Eq(oshape) {
		return b, nil
	}

	bdims := newShape.Dims()
	diff := adims - bdims
	if diff < 0 {
		return b, errors.Errorf("Unable to KeepDims for a result with shape %v. It has more dimensions than input %v", newShape, oshape)
	}
	var retShape tensor.Shape
	if expandLeft {
		retShape = tensor.BorrowInts(diff + newShape.Dims())
		for i := 0; i < diff; i++ {
			retShape[i] = 1
		}
		copy(retShape[diff:], newShape)
	} else {
		retShape = newShape.Clone()
		for i := 0; i < diff; i++ {
			retShape = append(retShape, 1)
		}

	}
	return Reshape(b, retShape)
}
