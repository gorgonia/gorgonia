package gorgonnx

import (
	"errors"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
)

func init() {
	register("Conv", newConv)
}

func newConv() operator {
	return &conv{}
}

// conv to be compatible with:
//    https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
// and
//    https://godoc.org/gorgonia.org/gorgonia#Conv2d
// test with go test -run=TestONNX/Conv
type conv struct {
	autopad     string
	pad         []int
	stride      []int
	dilation    []int
	group       int
	kernelShape tensor.Shape
}

func (c *conv) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	var err error
	if len(children) < 2 || len(children) > 3 {
		return errors.New("Conv: bad arity")
	}
	// autopadding needs to be applied now because it needs to be aware of the shape of the nodes
	switch c.autopad {
	case "NOTSET":
	case "":
	case "SAME_UPPER":
		for i, v := range children[0].gorgoniaNode.Shape()[2:] {
			outputD := ceilDivInt(v, c.stride[i])
			c.pad[i] = (outputD-1)*c.stride[i] + (c.kernelShape[i]-1)*c.dilation[i] + 1 - v
			if c.pad[i] < 0 {
				c.pad[i] = 0
			}
			if c.pad[i]%2 != 0 {
				return &onnx.ErrNotImplemented{
					Operator:       "conv",
					AttributeName:  "pads",
					AttributeValue: c.pad[i],
					Message:        "Asymetric padding",
				}
			}
			c.pad[i] /= 2
		}
	default:
		return &onnx.ErrNotImplemented{
			Operator: "Conv",
			Message:  "auto_pad " + c.autopad + " not implemented",
		}
	}
	convN, err := nnops.Conv2d(
		children[0].gorgoniaNode,
		children[1].gorgoniaNode,
		c.kernelShape,
		c.pad,
		c.stride,
		c.dilation)
	if err != nil {
		return &errOp{
			"conv",
			err,
		}
	}
	if len(children) == 3 {
		b, err := gorgonia.Reshape(children[2].gorgoniaNode, []int{1, children[2].gorgoniaNode.Shape()[0], 1, 1})
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
		convA, ba, err := gorgonia.Broadcast(convN, b, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
		n.gorgoniaNode, err = gorgonia.Add(convA, ba)
		if err != nil {
			return &errOp{
				"conv",
				err,
			}
		}
	} else {
		n.gorgoniaNode = convN
	}
	return nil
}

func (c *conv) init(o onnx.Operation) error {
	autoPad, ok := o.Attributes["auto_pad"]
	if ok {
		switch autoPad.(string) {
		case "NOTSET":
		case "VALID":
			c.pad = []int{0, 0}
		default:
			c.autopad = autoPad.(string)
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
	c.pad = []int{0, 0}
	pad, ok := o.Attributes["pads"]
	if ok {
		if pad, ok := pad.([]int64); ok {

			if len(pad) == 4 && (pad[0] != pad[1] || pad[2] != pad[3]) {
				return &onnx.ErrNotImplemented{
					Operator:       "Conv",
					AttributeName:  "pads",
					AttributeValue: pad,
					Message:        "Asymetric padding",
				}
			}

			if len(pad) == 4 {
				for i := 0; i < 2; i++ {
					c.pad[i] = int(pad[2*i])
				}
			} else if len(pad) == 2 {
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
	c.dilation = []int{1, 1}
	dilation, ok := o.Attributes["dilations"]
	if ok {
		if dilation, ok := dilation.([]int64); ok {
			c.dilation = make([]int, len(dilation))
			for i := 0; i < len(dilation); i++ {
				c.dilation[i] = int(dilation[i])
			}
		}
	}
	return nil
}
