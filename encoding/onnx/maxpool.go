package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func init() {
	register("MaxPool", newMaxpool)
}

func newMaxpool() operator {
	return &maxpool{}
}

// maxpool to be compatible with:
//    https://github.com/onnx/onnx/blob/master/docs/Operators.md#maxpool
// and
//    https://godoc.org/gorgonia.org/gorgonia#MaxPool2D
// test with go test -run=TestONNX/maxpool
type maxpool struct {
	padding     string
	dilation    []int
	pad         []int
	stride      []int
	kernelShape tensor.Shape
	ceilMode    bool
	divInt      func(a, b int) int
}

func (c *maxpool) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 1)
	if err != nil {
		return err
	}
	x := children[0].gorgoniaNode
	switch c.padding {
	case "SAME_UPPER":
		for i, v := range x.Shape()[2:] {
			j := 2 * i
			outputSpatialShape := ceilDivInt(v, c.stride[i])
			c.pad[j] = (outputSpatialShape-1)*c.stride[i] + ((c.kernelShape[i]-1)*c.dilation[i] + 1) - v
			if c.pad[j] < 0 {
				c.pad[j] = 0
			}
			if c.pad[j]%2 != 0 {
				c.pad[j] = c.pad[j]/2 + 1
				c.pad[j+1] = c.pad[j] - 1
			} else {
				c.pad[j] /= 2
				c.pad[j+1] = c.pad[j]
			}
		}
	default:
	}
	n.gorgoniaNode, err = gorgonia.MaxPool2D(
		children[0].gorgoniaNode,
		c.kernelShape,
		c.pad,
		c.stride)
	//c.ceilMode)

	return err
}

func (c *maxpool) init(o onnx.Operation) error {
	var autoPad string
	if a, ok := o.Attributes["auto_pad"]; ok {
		if autoPad, ok = a.(string); !ok {
			return errors.New("autopad is not a string")
		}

	}
	switch autoPad {
	case "NOTSET":
	case "SAME_UPPER":
		c.padding = autoPad
	case "":
	default:
		return &onnx.ErrNotImplemented{
			Operator: "maxpool",
			Message:  "auto_pad " + autoPad + " not implemented",
		}
	}
	kernelShape, ok := o.Attributes["kernel_shape"]
	if ok {
		if kernelShape, ok := kernelShape.([]int64); ok {
			c.kernelShape = make([]int, len(kernelShape))
			for i := 0; i < len(kernelShape); i++ {
				c.kernelShape[i] = int(kernelShape[i])
			}
		}
	}
	if len(c.kernelShape) != 2 {
		return &onnx.ErrNotImplemented{
			Operator: "maxpool",
			Message:  "maxpool for dim >2 not implemented",
		}
	}

	c.pad = []int{0, 0, 0, 0}
	pad, ok := o.Attributes["pads"]
	if ok {
    if pad, ok := pad.([]int64); ok {
			for i, v := range pad {
				c.pad[i] = int(v)
			}
			if len(pad) == 2 {
				for i := 0; i < 2; i++ {
					c.pad[i] = int(pad[i])
				}
			}
		}
	}
	c.stride = []int{1, 1}
	stride, ok := o.Attributes["strides"]
	if ok {
		if stride, ok := stride.([]int64); ok {
			if len(stride) == 4 {
				for i := 0; i < 2; i++ {
					c.stride[i] = int(stride[2*i])
				}
			} else if len(stride) == 2 {
				for i := 0; i < 2; i++ {
					c.stride[i] = int(stride[i])
				}
			}
		}
	}
	c.divInt = floorDivInt
	if ceilMode, ok := o.Attributes["ceil_mode"]; ok {
		if mode, ok := ceilMode.(int64); ok {
			if mode == 1 {
				c.ceilMode = true
				c.divInt = ceilDivInt
				return &onnx.ErrNotImplemented{
					Operator:       "maxpool",
					AttributeName:  "ceil_mode",
					AttributeValue: ceilMode,
					Message:        "ceil mode not implemented in Gorgonia (https://github.com/gorgonia/gorgonia/pull/294)",
				}
			}
		}
	}
	c.dilation = []int{1, 1}
	_, ok = o.Attributes["dilations"]
	if ok {
		return &onnx.ErrNotImplemented{
			Operator: "maxpool",
			Message:  "dilation not implemented",
		}
	}
	return nil
}
