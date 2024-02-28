package nnops

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

// computeShape computes the shape for the result. This should have been taken care of by the ShapeExpr so this method only exists in _test.go file.
func (op im2col[DT, T]) computeShape(s shapes.Shape) shapes.Shape {
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	h2 := (h+2*op.padH-(op.dilationH*(op.h-1)+1))/op.strideH + 1
	w2 := (w+2*op.padW-(op.dilationW*(op.w-1)+1))/op.strideW + 1
	c2 := c * op.h * op.w
	return shapes.Shape{b, h2, w2, c2}
}

var im2colTests = []struct {
	kernel   shapes.Shape
	pad      shapes.Shape
	stride   shapes.Shape
	dilation shapes.Shape

	willErr1 bool // willErr when calling Im2Col
}{
	{shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, false},
	{shapes.Shape{3, 3}, shapes.Shape{0, 0}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, false},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, false},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1}, shapes.Shape{2, 2}, shapes.Shape{1, 1}, false},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{3, 3}, false},
	{shapes.Shape{3, 3, 3}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{3, 3}, true},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1, 1}, shapes.Shape{1, 1}, shapes.Shape{3, 3}, true},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1}, shapes.Shape{1, 1, 1}, shapes.Shape{3, 3}, true},
	{shapes.Shape{3, 3}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{3, 3, 3}, true},
	{shapes.Shape{0, 0}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, true},
	{shapes.Shape{1, 1}, shapes.Shape{-1, -1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, true},
	{shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{0, 0}, shapes.Shape{1, 1}, true},
	{shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{1, 1}, shapes.Shape{0, 0}, true},
}

func TestIm2Col(t *testing.T) {
	assert := assert.New(t)
	for _, tc := range im2colTests {
		t.Run(fmt.Sprintf("kernel: %v, padding %v, stride: %v, dilation: %v", tc.kernel, tc.pad, tc.stride, tc.dilation), func(t *testing.T) {
			op, err := Im2Col[float64, *dense.Dense[float64]](tc.kernel, tc.pad, tc.stride, tc.dilation)
			if err != nil && !tc.willErr1 {
				t.Fatalf("Cannot create im2col with kernel %v pad %v, stride %v, dilation: %v", tc.kernel, tc.pad, tc.stride, tc.dilation)
			}
			if tc.willErr1 {
				return
			}
			d := dense.New[float64](tensor.WithShape(2, 1, 28, 28), tensor.WithBacking(tensor.Range[float64](0, 2*1*28*28)))
			o := op.(im2col[float64, *dense.Dense[float64]])
			outShape := o.computeShape(d.Shape())
			prealloc := dense.New[float64](tensor.WithShape(outShape...))
			retVal, err := op.PreallocDo(context.Background(), prealloc, d)
			if err != nil {
				t.Fatalf("Failed to run im2col with  kernel %v pad %v, stride %v, dilation: %v", tc.kernel, tc.pad, tc.stride, tc.dilation)
			}
			assert.Same(prealloc, retVal)

			retVal, err = op.Do(context.Background(), d)
			if err != nil {
				t.Fatalf("Failed to run im2col.Do with  kernel %v pad %v, stride %v, dilation: %v", tc.kernel, tc.pad, tc.stride, tc.dilation)
				return
			}
			assert.True(prealloc.Shape().Eq(retVal.Shape()))
			assert.Equal(prealloc.Data(), retVal.Data())
		})
	}
}
