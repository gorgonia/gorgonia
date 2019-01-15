package onnx

// START_CONV OMIT

// Conv operator
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv
//
// For more information about convolution, please visit https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
type Conv struct {
	KernelShape []int64 `attributeName:"kernel_shape" required:"true"`
	AutoPad     string  `attributeName:"auto_pad"`
	Dilations   []int64 `attributeName:"dilations"`
	Group       int64   `attributeName:"group"`
	Pads        []int64 `attributeName:"pads"`
	Strides     []int64 `attributeName:"strides"`
}

// END_CONV OMIT

/*
// Init the convolution operator
func (c *Conv) Init(attrs []*onnx.AttributeProto) error {
	c.name = "conv"
	type attributes struct {
		KernelShape []int64 `attributeName:"kernel_shape" required:"true"`
		AutoPad     string  `attributeName:"auto_pad"`
		Dilations   []int64 `attributeName:"dilations"`
		Group       int64   `attributeName:"group"`
		Pads        []int64 `attributeName:"pads"`
		Strides     []int64 `attributeName:"strides"`
	}
	// Set the default values
	attr := attributes{
		AutoPad:   "NOTSET",
		Group:     1,
		Strides:   []int64{1, 1},
		Pads:      []int64{0, 0},
		Dilations: []int64{1, 1},
	}
	err := onnx.UnmarshalAttributes(attrs, &attr)
	if err != nil {
		return err
	}
	// Set the obvious values
	c.KernelShape = int64ToInt(attr.KernelShape)
	c.Group = int(attr.Group)
	c.Strides = int64ToInt(attr.Strides)
	c.Dilations = int64ToInt(attr.Dilations)

	if c.Group != 1 {
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "group",
			AttributeValue: c.Group,
			Message:        "attribute not implemented for a value != 1",
		}
	}
	switch attr.AutoPad {
	case "NOTSET":
	case "VALID":
		c.Pads = []int{0, 0}
	case "SAME_UPPER":
		c.AutoPad = attr.AutoPad
	case "SAME_LOWER":
		c.AutoPad = attr.AutoPad
	default:
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "auto_pad",
			AttributeValue: attr.AutoPad,
			Message:        "Invalide value",
		}
	}

	if len(attr.Pads) == 4 && (attr.Pads[0] != attr.Pads[1] || attr.Pads[2] != attr.Pads[3]) {
		return &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "pads",
			AttributeValue: attr.Pads,
			Message:        "Asymetric padding",
		}
	}
	c.Pads = make([]int, 2)
	if len(attr.Pads) == 4 {
		for i := 0; i < 2; i++ {
			//c.Pads[i] = int(attr.Pads[2*i] + attr.Pads[2*i+1])
			c.Pads[i] = int(attr.Pads[2*i])
		}
	}
	return nil
}
*/

/*
// Apply ...
func (c *Conv) Apply(input ...*gorgonia.Node) ([]*gorgonia.Node, error) {
	if len(input) != 2 {
		return nil, &ErrBadArity{
			Operator:      "Conv",
			ExpectedInput: 2,
			ActualInput:   len(input),
		}
	}
	if len(input[1].Shape()) != 4 {
		return nil, &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "Kernel",
			AttributeValue: input[1].Shape(),
			Message:        "Kernel shape invalid",
		}

	}
	switch c.AutoPad {
	case "SAME_UPPER":
		outputHeight := int(math.Ceil(float64(input[0].Shape()[2]) / float64(c.Strides[0])))
		outputWidth := int(math.Ceil(float64(input[0].Shape()[3]) / float64(c.Strides[1])))
		c.Pads[0] = int(math.Max(float64((outputHeight-1)*c.Strides[0]+c.KernelShape[0]-input[0].Shape()[2]), float64(0))) / 2
		c.Pads[1] = int(math.Max(float64((outputWidth-1)*c.Strides[1]+c.KernelShape[1]-input[0].Shape()[3]), float64(0))) / 2
	case "SAME_LOWER":
		return nil, &onnx.ErrNotImplemented{
			Operator:       c.name,
			AttributeName:  "auto_pad",
			AttributeValue: c.AutoPad,
			Message:        "not supported",
		}
	default:
	}
	n, err := nnops.Conv2d(input[0], input[1], c.KernelShape, c.Pads, c.Strides, c.Dilations)
	return []*gorgonia.Node{n}, err
}
*/
