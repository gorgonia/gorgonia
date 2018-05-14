// +build cuda

package nnops

import (
	"fmt"
	"hash"
	"log"

	"github.com/chewxy/hm"
	"gorgonia.org/cu/dnn"
	"gorgonia.org/cu/dnn/interop"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var (
	_ G.Op       = &convolution{}
	_ G.CUDADoer = &convolution{}
)

type convolution struct {
	*cudnn.Convolution

	// created with these attributes
	padding, stride, dilation []int
	inShape, filterShape      tensor.Shape

	// cached descriptors
	xDesc, yDesc *cudnn.TensorDescriptor
	wDesc        *cudnn.Filter
}

func makeConvolutionOp(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *convolution, err error) {
	var xDesc *cudnn.TensorDescriptor
	var wDesc *cudnn.Filter
	if xDesc, err = t2cudnn.Describe(im); err != nil {
		return nil, err
	}
	if wDesc, err = t2cudnn.DescribeAsFilter(filter, cudnn.NCHW); err != nil {
		return nil, err
	}
	datatype := t2cudnn.Dtype2DataType(im.Dtype())
	conv, err := cudnn.NewConvolution(cudnn.DefaultMath, 1, pad, stride, dilation, cudnn.StandardConvolution, datatype)
	if err != nil {
		return nil, err
	}

	return &convolution{
		Convolution: conv,
		padding:     pad,
		stride:      stride,
		dilation:    dilation,

		inShape:     im.Shape().Clone(),
		filterShape: filter.Shape().Clone(),

		xDesc: xDesc,
		wDesc: wDesc,
	}, nil
}

func (c *convolution) Arity() int { return 2 }

func (c *convolution) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (c *convolution) InferShape(inputs ...G.DimSizer) (retVal tensor.Shape, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}
	return c.ForwardOutputShape(c.xDesc, c.wDesc, 2) //only conv2d is supported now
}

func (c *convolution) Do(inputs ...G.Value) (retVal G.Value, err error) {
	panic("not implemented")
}

func (c *convolution) ReturnsPtr() bool { return true }

func (c *convolution) CallsExtern() bool { return true }

func (c *convolution) OverwritesInput() int { return -1 }

func (c *convolution) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "Convolution:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convolution) Hashcode() uint32 { return simpleHash(c) }

func (c *convolution) String() string {
	return fmt.Sprintf("Convolution:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convolution) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}
	im, filter := inputs[0], inputs[1]

	if c.yDesc == nil {
		if c.yDesc, err = t2cudnn.Describe(prealloc.(tensor.Tensor)); err != nil {
			return
		}
	}

	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContext()

	if err = ctx.ConvolutionForward(1.0,
		c.xDesc, im.(cudnn.Memory),
		c.wDesc, filter.(cudnn.Memory),
		c.Convolution,
		cudnn.ConvolutionFwdAlgoImplicitGemm, nomem{},
		0, 1.0,
		c.yDesc, prealloc.(cudnn.Memory)); err != nil {
		return
	}
	return prealloc, nil
}

func (c *convolution) DoDiff(ctx G.ExecutionContext, inputs G.Nodes, output *G.Node) error {
	panic("not implemented")
}

func (c *convolution) DiffWRT(inputs int) []bool {
	return []bool{true, true}
}

func (c *convolution) SymDiff(inputs G.Nodes, output *G.Node, grad *G.Node) (retVal G.Nodes, err error) {
	log.Printf("Inputs %v, Output %v, Grad %v", inputs, output, grad)
	panic("not implemented")
}

type convolutionDiff struct {
	*convolution
}

func (c convolutionDiff) Arity() int { return 4 }

func (c convolutionDiff) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (c convolutionDiff) InferShape(...G.DimSizer) (tensor.Shape, error) {
	return c.inShape.Clone(), nil
}

func (c convolutionDiff) Do(...G.Value) (G.Value, error) {
	panic("not implemented")
}

func (c convolutionDiff) ReturnsPtr() bool { return true }

func (c convolutionDiff) CallsExtern() bool { return true }

func (c convolutionDiff) OverwritesInput() int { return -1 }

func (c convolutionDiff) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "ConvolutionDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c convolutionDiff) Hashcode() uint32 { return simpleHash(c) }

func (c convolutionDiff) String() string {
	return fmt.Sprintf("ConvolutionDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c convolutionDiff) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}
	im, filter, output, grad := inputs[0], inputs[1], inputs[2], inputs[3]

	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContext()

	if err = ctx.ConvolutionForward(1.0,
		c.xDesc, im.(cudnn.Memory),
		c.wDesc, filter.(cudnn.Memory),
		c.Convolution,
		cudnn.ConvolutionFwdAlgoImplicitGemm, nomem{},
		0, 1.0,
		c.yDesc, prealloc.(cudnn.Memory)); err != nil {
		return
	}
	return prealloc, nil
}
