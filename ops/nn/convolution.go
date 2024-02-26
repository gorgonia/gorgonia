//go:build !cuda
// +build !cuda

package nnops

import (
	"context"
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// convolution represents a convolution operation. It is used to store the parameters of the convolution operation.
type convolution[DT any, T values.Value[DT]] struct {
	padding, stride, dilation []int
	inShape, filterShape      tensor.Shape
}

func NewConvolution[DT any, T values.Value[DT]](padding, stride, dilation []int, inShape, filterShape tensor.Shape) *convolution[DT, T] {
	return &convolution[DT, T]{
		padding:     padding,
		stride:      stride,
		dilation:    dilation,
		inShape:     inShape,
		filterShape: filterShape,
	}
}

func (op *convolution[DT, T]) Arity() int {
	return 2
}

func (op *convolution[DT, T]) Type() hm.Type {
	return "convolution"
}

func (op *convolution[DT, T]) ShapeExpr() shapes.Expr {
	return shapes.Arbitrary
}

func (op *convolution[DT, T]) String() string {
	return fmt.Sprintf("convolution{padding: %v, stride: %v, dilation: %v, inShape: %v, filterShape: %v}", op.padding, op.stride, op.dilation, op.inShape, op.filterShape)

}

// Do performs the convolution operation.
func (c *convolution[DT, T]) Do(ctx context.Context, vs ...T) (retVal T, err error) {

}

type convDiffIm[DT any, T values.Value[DT]] struct {
	*convolution[DT, T]
}
