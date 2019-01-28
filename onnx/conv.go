package onnx

import (
	"errors"
	"math"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
	"gorgonia.org/tensor"
)

// START_CONV OMIT

// Conv2d operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
//
// For more information about convolution, please visit https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
type Conv2d struct {
	KernelShape []int64 `attributeName:"kernel_shape" required:"true"`
	AutoPad     string  `attributeName:"auto_pad"`
	Dilations   []int64 `attributeName:"dilations"`
	Group       int64   `attributeName:"group"`
	Pads        []int64 `attributeName:"pads"`
	Strides     []int64 `attributeName:"strides"`
	// END_CONV OMIT
	kernelShape tensor.Shape
	pad         []int
	stride      []int
	dilation    []int
}

// NewConv operator with default values ...
func NewConv() *Conv2d {
	return &Conv2d{
		AutoPad:   "NOTSET",
		Group:     1,
		Strides:   []int64{1, 1},
		Pads:      []int64{0, 0},
		Dilations: []int64{1, 1},
	}
}

// Constructor for the Convolution operator
func (c *Conv2d) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
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
		if c.Group != 1 {
			return nil, errors.New("Not implemented")
		}
		if len(c.Pads) == 4 && (c.Pads[0] != c.Pads[1] || c.Pads[2] != c.Pads[3]) {
			return nil, errors.New("Not implemented")
		}
		c.pad = make([]int, 2)
		if len(c.Pads) == 4 {
			for i := 0; i < 2; i++ {
				c.pad[i] = int(c.Pads[2*i])
			}
		}

		switch c.AutoPad {
		case "VALID":
			c.pad = []int{0, 0}
		case "SAME_UPPER":
			outputHeight := int64(
				math.Ceil(
					float64(children[0].Shape()[2]) /
						float64(c.Strides[0])))
			outputWidth := int64(
				math.Ceil(
					float64(children[0].Shape()[3]) /
						float64(c.Strides[1])))
			c.pad[0] = int(
				math.Max(
					float64((outputHeight-1)*c.Strides[0]+
						c.KernelShape[0]-
						int64(children[0].Shape()[2])),
					float64(0))) /
				2
			c.pad[1] = int(
				math.Max(
					float64((outputWidth-1)*c.Strides[1]+
						c.KernelShape[1]-
						int64(children[0].Shape()[3])),
					float64(0))) /
				2
		case "SAME_LOWER":
			return nil, errors.New("Not Implemented")
		default:
		}
		c.stride = make([]int, len(c.Strides))
		for i, v := range c.Strides {
			c.stride[i] = int(v)
		}
		c.dilation = make([]int, len(c.Dilations))
		for i, v := range c.Dilations {
			c.dilation[i] = int(v)
		}
		c.kernelShape = make([]int, len(c.KernelShape))
		for i, v := range c.KernelShape {
			c.kernelShape[i] = int(v)
		}

		ops, err := engine.NewConv2d(c.kernelShape, c.pad, c.stride, c.dilation)(g, n.(*engine.Node))
		return ops, err
	}
}
