package engine

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/distro"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
	"gorgonia.org/tensor"
)

// BinaryXent is a convenience function for doing binary crossentropy stuff.
// The formula is as below:
// 		-(y * logprob) +  (1-y)(1-logprob)
func BinaryXent(output, target *Node) (retVal *Node, err error) {
	var one *Node
	var logO, omt, omo, tLogO *Node

	// which constant one to use?
	var dt tensor.Dtype
	if dt, err = dtypeOf(output.t); err != nil {
		return nil, errors.Wrapf(err, dtypeExtractionFail, output.t)
	}

	switch dt {
	case Float64:
		one = onef64(target.g)
	case Float32:
		one = onef32(target.g)
	default:
		return nil, errors.Errorf(nyiFail, "BinaryXEnt", dt)
	}

	if logO, err = Log(output); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if omt, err = Sub(one, target); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if omo, err = Sub(one, output); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if tLogO, err = HadamardProd(target, logO); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if retVal, err = Log(omo); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if retVal, err = HadamardProd(omt, retVal); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if retVal, err = Add(tLogO, retVal); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	return Neg(retVal)
}

// Dropout is a convenience function to implement dropout.
// It uses randomly zeroes out a *Tensor with a probability drawn from
// a uniform distribution
func Dropout(x *Node, prob float64) (retVal *Node, err error) {
	g := x.g
	if prob == 0.0 {
		return x, nil
	}

	var dt tensor.Dtype
	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	var opp, pr value.Value // opp = 1 per p
	switch dt {
	case Float64:
		opp, _ = value.AnyToScalar(1.0 / prob)
		pr, _ = value.AnyToScalar(prob)
	case Float32:
		opp, _ = value.AnyToScalar(float32(1.0 / prob))
		pr, _ = value.AnyToScalar(float32(prob))
	default:
		return nil, errors.Errorf(nyiTypeFail, "Dropout()", dt)
	}

	p := g.NewConstant(pr)
	g.AddNode(p)

	c := g.NewConstant(opp)
	g.AddNode(c)

	m := UniformRandomNode(x.g, dt, 0, 1, x.shape...)
	if retVal, err = Gt(m, p, true); err != nil {
		return nil, errors.Wrap(err, "Greater Than failed")
	}

	if retVal, err = HadamardProd(x, retVal); err != nil {
		return nil, errors.Wrap(err, mulFail)
	}

	return HadamardDiv(retVal, c)
}

// Rectify is a convenience function for creating rectified linear units activation functions.
// This function uses >=, which is the canonical version. If you want to use >, you can create
// your own by just following this.
func Rectify(x *Node) (retVal *Node, err error) {
	var zero *Node
	var dt tensor.Dtype

	// which zero to use?
	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}
	switch dt {
	case Float64:
		zero = zerof64(x.g)
	case Float32:
		zero = zerof32(x.g)
	default:
		return nil, errors.Errorf(nyiFail, "ReLu", dt)
	}

	cmp := newElemBinOp(gteOpType, x, zero)
	cmp.retSame = true

	if retVal, err = ApplyOp(cmp, x, zero); err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}

	return HadamardProd(x, retVal)
}

// NewIm2Col converts a BCHW image block to columns. The kernel, pad and stride parameter must be shape of size 2, no more no less
// This poor naming scheme clearly comes from matlab
func NewIm2Col(kernel, pad, stride, dilation tensor.Shape) Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		if kernel.Dims() != 2 {
			return nil, errors.Errorf("kernel shape is supposed to have a dim of 2")
		}
		if pad.Dims() != 2 {
			return nil, errors.Errorf("pad is supposed to have a dim of 2")
		}
		if stride.Dims() != 2 {
			return nil, errors.Errorf("strides is supposed to have a dim of 2")
		}
		if dilation.Dims() != 2 {
			return nil, errors.Errorf("dilation is supposed to have a dim of 2")
		}

		if kernel[0] <= 0 || kernel[1] <= 0 {
			return nil, errors.Errorf("cannot have negative or 0 in kernel shape")
		}

		if stride[0] <= 0 || stride[1] <= 0 {
			return nil, errors.Errorf("cannot have negative or 0 in stride: %v", stride)
		}

		if pad[0] < 0 || pad[1] < 0 {
			return nil, errors.Errorf("cannot have negative padding")
		}

		if dilation[0] <= 0 || dilation[1] <= 0 {
			return nil, errors.Errorf("canot have negative or 0 in dilation. %v", dilation)
		}

		return makeIm2ColOp(kernel[0], kernel[1], pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1]), nil
	}
}

// NewConv2d returns a simple 2D convoution, to be used for CPU computation only. If CuDNN is used, use the CUDAConv2D function.
// These are the properties the inputs must fulfil:
//
// im: must have 4D shape. Expected format is BCHW (batch, channel, height, width)
// filter: must have 4D shape: (batch, kernel, height, width)
// kernelShape: shape of the filter kernel
// pad: len(pad) == 2
// stride: len(stride) == 2
// dilation: len(dilation) == 2
func NewConv2d(kernelShape tensor.Shape, pad, stride, dilation []int) Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		it := getOrderedChildren(g, n)
		if it.Len() != 2 {
			return nil, errors.New("Unexpected number of children")
		}
		children := make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}
		im := children[0]
		filter := children[1]
		// niceness for defaults
		if pad == nil {
			pad = []int{0, 0}
		}
		if dilation == nil {
			dilation = []int{1, 1}
		}

		// checks
		for _, s := range stride {
			if s <= 0 {
				return nil, errors.Errorf("Cannot use strides of less than or equal 0: %v", stride)
			}
		}

		for _, p := range pad {
			if p < 0 {
				return nil, errors.Errorf("Cannot use padding of less than 0: %v", pad)
			}
		}

		for _, d := range dilation {
			if d <= 0 {
				return nil, errors.Errorf("Cannot use dilation less than or eq 0 %v", dilation)
			}
		}
		// check if the graph is a weighted builder
		builder, ok := g.(graph.DirectedWeightedBuilder)
		if !ok {
			return nil, errors.Errorf("Conv2d needs to modify the graph but is not a DirectedWeightedBuilder")
		}
		_, ok = g.(graph.EdgeRemover)
		if !ok {
			return nil, errors.Errorf("Conv2d needs to modify the graph but is not a DirectedWeightedBuilder")
		}
		// Create the node that will receive the result of im2col
		colIm := builder.NewNode().(*Node)
		builder.AddNode(colIm)
		// Link it to the input tensor
		builder.SetWeightedEdge(builder.NewWeightedEdge(colIm, im, 0.0))

		err := g.(*ExprGraph).ApplyOp(NewIm2Col(kernelShape, pad, stride, dilation), colIm)
		if err != nil {
			return nil, err
		}

		layer := filter.Shape()[0]
		kernel := filter.Shape()[1]
		row := filter.Shape()[2]
		col := filter.Shape()[3]

		// Create the node that will receive the result of the reshape of the filter
		flattened := builder.NewNode().(*Node)
		builder.AddNode(flattened)
		// Link it to the input tensor
		builder.SetWeightedEdge(builder.NewWeightedEdge(flattened, filter, 0.0))

		err = g.(*ExprGraph).ApplyOp(NewReshapeOperation(tensor.Shape{layer, kernel * row * col}), flattened)
		if err != nil {
			return nil, err
		}

		// extract patch
		batch := colIm.Shape()[0]
		m := colIm.Shape()[1]
		nn := colIm.Shape()[2]
		z := colIm.Shape()[3]

		// Create the nodes for patch and colImLayer
		patch := builder.NewNode().(*Node)
		builder.AddNode(patch)
		// Link it to the input tensor
		builder.SetWeightedEdge(builder.NewWeightedEdge(patch, colIm, 0.0))

		colImLayer := builder.NewNode().(*Node)
		builder.AddNode(colImLayer)
		// Link it to the input tensor
		builder.SetWeightedEdge(builder.NewWeightedEdge(colImLayer, patch, 0.0))

		err = g.(*ExprGraph).ApplyOp(NewReshapeOperation(tensor.Shape{batch * m * nn, z}), patch)
		if err != nil {
			return nil, err
		}

		op := linAlgBinOp{
			āBinaryOperator: matMulOperator,
			transA:          false,
			transB:          true,
		}

		err = g.(*ExprGraph).ApplyOp(newLinAlgBinOperation(op), colImLayer)
		if err != nil {
			return nil, err
		}

		// now reshape and transpose the values back into the original order
		res := builder.NewNode().(*Node)
		builder.AddNode(res)
		// Link it to the input tensor
		builder.SetWeightedEdge(builder.NewWeightedEdge(res, colImLayer, 0.0))

		err = g.(*ExprGraph).ApplyOp(NewReshapeOperation(tensor.Shape{batch, m, nn, layer}), res)
		if err != nil {
			return nil, err
		}

		// Now remove the original links from n -> children[0] and n -> children[1]
		g.(graph.EdgeRemover).RemoveEdge(n.ID(), filter.ID())
		g.(graph.EdgeRemover).RemoveEdge(n.ID(), im.ID())
		// And create the new links
		builder.SetWeightedEdge(builder.NewWeightedEdge(n, res, 0.0))

		return NewTransposeOperation(0, 3, 1, 2)(g, n)
	}
}

/*
// Conv2d ...
func Conv2d(im, filter *Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *Node, err error) {
	// niceness for defaults
	if pad == nil {
		pad = []int{0, 0}
	}
	if dilation == nil {
		dilation = []int{1, 1}
	}

	// checks
	for _, s := range stride {
		if s <= 0 {
			return nil, errors.Errorf("Cannot use strides of less than or equal 0: %v", stride)
		}
	}

	for _, p := range pad {
		if p < 0 {
			return nil, errors.Errorf("Cannot use padding of less than 0: %v", pad)
		}
	}

	for _, d := range dilation {
		if d <= 0 {
			return nil, errors.Errorf("Cannot use dilation less than or eq 0 %v", dilation)
		}
	}

	var colIm *Node
	if colIm, err = Im2Col(im, kernelShape, pad, stride, dilation); err != nil {
		return
	}

	layer := filter.Shape()[0]
	kernel := filter.Shape()[1]
	row := filter.Shape()[2]
	col := filter.Shape()[3]

	var flattened *Node
	if flattened, err = Reshape(filter, tensor.Shape{layer, kernel * row * col}); err != nil {
		return
	}

	// extract patch
	batch := colIm.Shape()[0]
	m := colIm.Shape()[1]
	n := colIm.Shape()[2]
	z := colIm.Shape()[3]

	var patch, colImLayer *Node
	if patch, err = Reshape(colIm, tensor.Shape{batch * m * n, z}); err != nil {
		return
	}

	op := linAlgBinOp{
		āBinaryOperator: matMulOperator,
		transA:          false,
		transB:          true,
	}

	if colImLayer, err = ApplyOp(op, patch, flattened); err != nil {
		return
	}

	// now reshape and transpose the values back into the original order
	var res *Node
	if res, err = Reshape(colImLayer, tensor.Shape{batch, m, n, layer}); err != nil {
		return
	}
	return Transpose(res, 0, 3, 1, 2)
}
*/

/*
// Conv1d is a 1D convlution. It relies on Conv2D
func Conv1d(in, filter *Node, kernel, pad, stride, dilation int) (*Node, error) {
	return Conv2d(in, filter, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride}, []int{1, dilation})
}
*/

// MaxPool2D ...
func MaxPool2D(x *Node, kernel tensor.Shape, pad, stride []int) (*Node, error) {
	xShape := x.Shape()
	h, w := xShape[2], xShape[3]
	kh, kw := kernel[0], kernel[1]
	ph, pw := pad[0], pad[1]

	// check shape
	if xShape.Dims() != 4 {
		return nil, errors.Errorf("Expected input to have a shape with dimension 4")
	}
	if kernel.Dims() != 2 {
		return nil, errors.Errorf("Expected kernel to have a shape of dimension 2")
	}

	if h-kh == 0 && ph == 0 {
		// error
		return nil, errors.New("Impossible height/kernel/pad combination")
	}

	if w-kw == 0 && pw == 0 {
		// error
		return nil, errors.New("Impossible width/kernel/pad combination")
	}

	op := newMaxPoolOp(xShape, kernel, pad, stride)
	return ApplyOp(op, x)
}

// BatchNorm ...
func BatchNorm(x, scale, bias *Node, momentum, epsilon float64) (retVal, γ, β *Node, op *BatchNormOp, err error) {
	dt, err := dtypeOf(x.Type())
	if err != nil {
		return nil, nil, nil, nil, err
	}
	batches := x.Shape()[0]
	channels := x.Shape()[1]
	spatialDim := x.Shape().TotalSize() / (channels * batches)

	mean := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	variance := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	ma := tensor.New(tensor.Of(dt), tensor.WithShape(1))

	meanT := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	varianceT := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	tmp := tensor.New(tensor.Of(dt), tensor.WithShape(x.Shape().Clone()...))
	xNorm := tensor.New(tensor.Of(dt), tensor.WithShape(x.Shape().Clone()...))
	batchSumMultiplier := tensor.New(tensor.Of(dt), tensor.WithShape(batches))

	var uno interface{}
	switch dt {
	case Float64:
		uno = float64(1)
	case Float32:
		uno = float32(1)
	}
	spatialSumMultiplier := tensor.New(tensor.Of(dt), tensor.WithShape(spatialDim))
	if err = spatialSumMultiplier.Memset(uno); err != nil {
		return nil, nil, nil, nil, err
	}

	numByChans := tensor.New(tensor.Of(dt), tensor.WithShape(channels*batches))
	if err = batchSumMultiplier.Memset(uno); err != nil {
		return nil, nil, nil, nil, err
	}

	op = &BatchNormOp{
		momentum: momentum,
		epsilon:  epsilon,

		mean:     mean,
		variance: variance,
		ma:       ma,

		meanT:                meanT,
		varianceT:            varianceT,
		tmpT:                 tmp,
		xNorm:                xNorm,
		batchSumMultiplier:   batchSumMultiplier,
		numByChans:           numByChans,
		spatialSumMultiplier: spatialSumMultiplier,
		training:             true,
	}
	g := x.Graph()
	dims := x.Shape().Dims()

	if scale == nil {
		scale = g.NewTensor(dt, dims, WithShape(x.Shape().Clone()...), WithName(x.Name()+"_γ"), WithInit(distro.GlorotN(1.0)))
		g.AddNode(scale)
	}
	if bias == nil {
		bias = g.NewTensor(dt, dims, WithShape(x.Shape().Clone()...), WithName(x.Name()+"_β"), WithInit(distro.GlorotN(1.0)))
		g.AddNode(bias)
	}

	if retVal, err = ApplyOp(op, x); err != nil {
		return nil, nil, nil, nil, err
	}
	if retVal, err = HadamardProd(scale, retVal); err != nil {
		return nil, nil, nil, nil, err
	}
	retVal, err = Add(retVal, bias)

	return retVal, scale, bias, op, err
}
