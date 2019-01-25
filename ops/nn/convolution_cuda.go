// +build cuda

package nnops

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	cudnn "gorgonia.org/cu/dnn"
	t2cudnn "gorgonia.org/cu/dnn/interop"
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
	machine.Engines()[int(dev)].DoWork()
	ctx := machine.CUDNNContexts()[int(dev)]

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
	var outDesc *cudnn.TensorDescriptor
	if outDesc, err = t2cudnn.Describe(output); err != nil {
		return nil, err
	}
	diffIm := &convDiffIm{
		convolution: c,
		outputDesc:  outDesc,
	}
	diffFilter := &convDiffFilter{
		convolution: c,
		outputDesc:  outDesc,
	}

	retVal = make(G.Nodes, 2)
	if retVal[0], err = G.ApplyOp(diffIm, inputs[0], grad); err != nil {
		return nil, err
	}
	if retVal[1], err = G.ApplyOp(diffFilter, inputs[1], grad); err != nil {
		return nil, err
	}

	return
}

// convDiffIm is the d(z)/d(im) operation. See also convDiffFilter
type convDiffIm struct {
	*convolution
	outputDesc *cudnn.TensorDescriptor
}

func (c *convDiffIm) Arity() int { return 2 }

func (c *convDiffIm) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (c *convDiffIm) InferShape(shps ...G.DimSizer) (tensor.Shape, error) {
	return c.inShape.Clone(), nil
}

func (c *convDiffIm) Do(...G.Value) (G.Value, error) {
	panic("not implemented")
}

func (c *convDiffIm) ReturnsPtr() bool { return true }

func (c *convDiffIm) CallsExtern() bool { return true }

func (c *convDiffIm) OverwritesInput() int { return -1 }

func (c *convDiffIm) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "ConvolutionImDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convDiffIm) Hashcode() uint32 { return simpleHash(c) }

func (c *convDiffIm) String() string {
	return fmt.Sprintf("ConvolutionImDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convDiffIm) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}
	filter, grad := inputs[0], inputs[1]

	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContexts()[int(dev)]

	if err = ctx.ConvolutionBackwardData(1.0,
		c.wDesc, filter.(cudnn.Memory),
		c.outputDesc, grad.(cudnn.Memory),
		c.Convolution,
		cudnn.ConvolutionBwdDataAlgo0, nomem{},
		0, 1.0,
		c.xDesc, prealloc.(cudnn.Memory)); err != nil {
		return
	}
	return prealloc, nil
}

type convDiffFilter struct {
	*convolution                         // shared struct as convDiffIm
	outputDesc   *cudnn.TensorDescriptor // shared output descriptor with convDiffIm
}

func (c *convDiffFilter) Arity() int { return 2 }

func (c *convDiffFilter) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (c *convDiffFilter) InferShape(...G.DimSizer) (tensor.Shape, error) {
	return c.filterShape.Clone(), nil
}

func (c *convDiffFilter) Do(...G.Value) (G.Value, error) {
	panic("not implemented")
}

func (c *convDiffFilter) ReturnsPtr() bool { return true }

func (c *convDiffFilter) CallsExtern() bool { return true }

func (c *convDiffFilter) OverwritesInput() int { return -1 }

func (c *convDiffFilter) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "ConvolutionFilterDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convDiffFilter) Hashcode() uint32 { return simpleHash(c) }

func (c *convDiffFilter) String() string {
	return fmt.Sprintf("ConvolutionFilterDiff:%v-%v-%v", c.Padding(), c.FilterStride(), c.Dilation())
}

func (c *convDiffFilter) CUDADo(extern G.External, dev G.Device, prealloc G.Value, inputs ...G.Value) (retVal G.Value, err error) {
	if err = checkArity(c, len(inputs)); err != nil {
		return
	}
	im, grad := inputs[0], inputs[1]

	machine := extern.(G.CUDAMachine)
	ctx := machine.CUDNNContexts()[int(dev)]

	if err = ctx.ConvolutionBackwardFilter(1.0,
		c.xDesc, im.(cudnn.Memory),
		c.outputDesc, grad.(cudnn.Memory),
		c.Convolution,
		cudnn.ConvolutionBwdFilterAlgo0, nomem{},
		0, 1.0,
		c.wDesc, prealloc.(cudnn.Memory)); err != nil {
		return
	}
	return prealloc, nil
}
