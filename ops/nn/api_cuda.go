// +build cuda

package nnops

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func Conv2d(im, filter *G.Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *G.Node, err error) {
	var op *convolution
	if op, err = makeConvolutionOp(im, filter, kernelShape, pad, stride, dilation); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, im, filter)
}

func Conv1d(in, filter *G.Node, kernel, pad, stride, dilation int) (*G.Node, error) {
	return Conv2d(in, filter, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride}, []int{1, dilation})
}

func MaxPool2D(x *G.Node, kernel tensor.Shape, pad, stride []int) (retVal *G.Node, err error) {
	var op *maxpool
	if op, err = newMaxPoolOp(x, kernel, pad, stride); err != nil {
		return nil, err
	}
	return G.ApplyOp(op, x)
}

func Dropout(x *G.Node, prob float64) (retVal *G.Node, err error) {
	var op *dropout
	if op, err = newDropout(x, prob); err != nil {
		return nil, err
	}

	// states := &scratchOp{x.Shape().Clone(), x.Dtype(), ""}
	// m := G.NewUniqueNode(G.WithType(x.Type()), G.WithOp(states), G.In(x.Graph()), G.WithShape(states.shape...))

	retVal, err = G.ApplyOp(op, x)
	return
}

func Rectify(x *G.Node) (retVal *G.Node, err error) {
	var op *activation
	if op, err = newRelu(); err != nil {
		return nil, err
	}
	retVal, err = G.ApplyOp(op, x)
	return
}

func BatchNorm(x, scale, bias *G.Node, momentum, epsilon float64) (retVal, γ, β *G.Node, op *BatchNormOp, err error) {
	dt, err := dtypeOf(x.Type())
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// batches := x.Shape()[0]
	channels := x.Shape()[1]
	H, W := x.Shape()[2], x.Shape()[3]
	// spatialDim := x.Shape().TotalSize() / (channels * batches)
	scratchShape := tensor.Shape{1, channels, H, W}

	// scaleScratch := &scratchOp{x.Shape().Clone(), dt, "scale"}
	// biasScratch := &scratchOp{x.Shape().Clone(), dt, "bias"}
	meanScratch := &gpuScratchOp{scratchOp{x.Shape().Clone(), dt, "mean"}}
	varianceScratch := &gpuScratchOp{scratchOp{x.Shape().Clone(), dt, "variance"}}
	cacheMeanScratch := &gpuScratchOp{scratchOp{scratchShape, dt, "cacheMean"}}
	cacheVarianceScratch := &gpuScratchOp{scratchOp{scratchShape, dt, "cacheVariance"}}

	g := x.Graph()
	dims := len(x.Shape())
	mean := G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithName(x.Name()+"_mean"), G.WithOp(meanScratch))
	variance := G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithName(x.Name()+"_variance"), G.WithOp(varianceScratch))
	cacheMean := G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithOp(cacheMeanScratch))
	cacheVariance := G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithOp(cacheVarianceScratch))

	if scale == nil {
		scale = G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithName(x.Name()+"_γ"), G.WithInit(G.GlorotN(1.0)))
	}

	if bias == nil {
		bias = G.NewTensor(g, dt, dims, G.WithShape(scratchShape.Clone()...), G.WithName(x.Name()+"_β"), G.WithInit(G.GlorotN(1.0)))
	}

	op = newBatchNormOp(momentum, epsilon)
	retVal, err = G.ApplyOp(op, x, scale, bias, mean, variance, cacheMean, cacheVariance)
	return retVal, scale, bias, op, err
}
