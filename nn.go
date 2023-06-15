package gorgonia

import (
	"fmt"
	"time"

	rng "github.com/leesper/go_rng"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
)

// BinaryXent is a convenience function for doing binary crossentropy stuff.
// The formula is as below:
// 		-(y * log(prob)) - (1-y)log(1-prob)
func BinaryXent(output, target *Node) (retVal *Node, err error) {
	var one, oneMore *Node
	var logO, omt, omo, tLogO *Node

	// which constant one to use?
	var dt tensor.Dtype
	if dt, err = dtypeOf(output.t); err != nil {
		return nil, errors.Wrapf(err, dtypeExtractionFail, output.t)
	}

	switch dt {
	case Float64:
		one = onef64
		oneMore = oneMoref64
	case Float32:
		one = onef32
		oneMore = oneMoref32
	default:
		return nil, errors.Errorf(nyiFail, "BinaryXEnt", dt)
	}

	if logO, err = Log(output); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if omt, err = Sub(one, target); err != nil {
		return nil, errors.Wrap(err, operationError)
	}

	if omo, err = Sub(oneMore, output); err != nil {
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
func Dropout(x *Node, dropProb float64) (retVal *Node, err error) {
	rand := rng.NewUniformGenerator(time.Now().UnixNano())

	op := newDropoutOp(dropProb, func() float64 { return rand.Float64Range(0, 1) })

	return ApplyOp(op, x)
}

// LeakyRelu returns a node whose underlying value is:
//   f(x) = alpha * x if x < 0
//   f(x) = x for x ⩾ 0
// applied elementwise.
func LeakyRelu(x *Node, alpha float64) (*Node, error) {
	var zero *Node
	var dt tensor.Dtype
	var err error
	var alphaN *Node

	// which zero to use?
	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}
	switch dt {
	case Float64:
		zero = zerof64
		alphaN = NewConstant(alpha)
	case Float32:
		zero = zerof32
		alphaN = NewConstant(float32(alpha))
	default:
		return nil, errors.Errorf(nyiFail, "ReLu", dt)
	}

	gteZeroOp := newElemBinOp(gteOpType, x, zero)
	gteZeroOp.retSame = true

	xGteZeroCmp, err := ApplyOp(gteZeroOp, x, zero)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	ltZeroOp := newElemBinOp(ltOpType, x, zero)
	ltZeroOp.retSame = true

	xLtZeroCmp, err := ApplyOp(ltZeroOp, x, zero)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	xGteZero, err := HadamardProd(x, xGteZeroCmp)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	xLtZero, err := HadamardProd(x, xLtZeroCmp)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	xLtZeroAlpha, err := HadamardProd(xLtZero, alphaN)
	if err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	return Add(xGteZero, xLtZeroAlpha)
}

// Rectify is a convenience function for creating rectified linear units activation functions.
// This function uses ⩾, which is the canonical version. If you want to use >, you can create
// your own by just following this.
func Rectify(x *Node) (retVal *Node, err error) {
	var zero *Node
	var dt tensor.Dtype
	group := encoding.NewGroup("Rectify")

	// which zero to use?
	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}
	switch dt {
	case Float64:
		zero = zerof64
	case Float32:
		zero = zerof32
	default:
		return nil, errors.Errorf(nyiFail, "ReLu", dt)
	}

	cmp := newElemBinOp(gteOpType, x, zero)
	cmp.retSame = true

	if retVal, err = ApplyOp(cmp, x, zero); err != nil {
		return nil, errors.Wrap(err, applyOpFail)
	}
	retVal.groups = retVal.groups.Upsert(group)

	return HadamardProd(x, retVal)
}

// Im2Col converts a BCHW image block to columns. The kernel, pad and stride parameter must be shape of size 2, no more no less
// This poor naming scheme clearly comes from matlab
func Im2Col(n *Node, kernel, pad, stride, dilation tensor.Shape) (retVal *Node, err error) {
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
		return nil, errors.Errorf("cannot have negative or 0 in dilation. %v", dilation)
	}

	op := makeIm2ColOp(kernel[0], kernel[1], pad[0], pad[1], stride[0], stride[1], dilation[0], dilation[1])
	return ApplyOp(op, n)
}

// Conv2d is a simple 2D convolution, to be used for CPU computation only.
// If CuDNN is used, use the CUDAConv2D function.
// These are the properties the inputs must fulfil:
//
// - im: must have 4D shape. Expected format is BCHW (batch, channels, height, width)
// - filter: must have 4D shape: (batch, kernel, height, width)
// - kernelShape: shape of the filter kernel
// - pad: len(pad) == 2, defaults to []int{0, 0} if nil is passed
// - stride: len(stride) == 2, example: []int{1, 1}
// - dilation: len(dilation) == 2, defaults to []int{1, 1} if nil is passed
func Conv2d(im, filter *Node, kernelShape tensor.Shape, pad, stride, dilation []int) (retVal *Node, err error) {
	group := encoding.NewGroup("Convolution")
	// niceness for defaults
	if pad == nil {
		pad = []int{0, 0}
	}
	if dilation == nil {
		dilation = []int{1, 1}
	}

	if im.Shape().Dims() != 4 {
		return nil, fmt.Errorf("im should have 4 dims, got %v dims", im.Shape().Dims())
	}

	if filter.Shape().Dims() != 4 {
		return nil, fmt.Errorf("filter should have 4 dims, got %v dims", filter.Shape().Dims())
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
		return nil, fmt.Errorf("Im2Col to failed: %w", err)
	}
	colIm.groups = colIm.groups.Upsert(group)

	layer := filter.Shape()[0]
	kernel := filter.Shape()[1]
	row := filter.Shape()[2]
	col := filter.Shape()[3]

	if colIm.Shape()[3] != kernel*row*col {
		return nil, fmt.Errorf("%d (kernel) * %d (width) * %d (height) must be %d, got %d", kernel, row, col, colIm.Shape()[3], kernel*row*col)
	}

	var flattened *Node
	if flattened, err = Reshape(filter, tensor.Shape{layer, kernel * row * col}); err != nil {
		return nil, fmt.Errorf("reshaping filter from %v to (%v, %v * %v * %v) failed: %w", filter.Shape(), layer, kernel, row, col, err)
	}
	flattened.groups = flattened.groups.Upsert(group)

	// extract patch
	batch := colIm.Shape()[0]
	m := colIm.Shape()[1]
	n := colIm.Shape()[2]
	z := colIm.Shape()[3]

	var patch, colImLayer *Node
	if patch, err = Reshape(colIm, tensor.Shape{batch * m * n, z}); err != nil {
		return nil, fmt.Errorf("reshaping colIm from %v to (%v * %v * %v * %v) failed: %w", colIm.Shape(), batch, m, n, z, err)
	}
	patch.groups = patch.groups.Upsert(group)

	op := linAlgBinOp{
		āBinaryOperator: matMulOperator,
		transA:          false,
		transB:          true,
	}

	if colImLayer, err = ApplyOp(op, patch, flattened); err != nil {
		return nil, fmt.Errorf("failed to apply op: %w", err)
	}
	colImLayer.groups = colImLayer.groups.Upsert(group)

	// now reshape and transpose the values back into the original order
	var res *Node
	if res, err = Reshape(colImLayer, tensor.Shape{batch, m, n, layer}); err != nil {
		return nil, fmt.Errorf("failed to reshape %v to (%v, %v, %v, %v): %w", colImLayer.Shape(), batch, m, n, layer, err)
	}
	res.groups = res.groups.Upsert(group)
	ret, err := Transpose(res, 0, 3, 1, 2)
	if err != nil {
		return nil, fmt.Errorf("transpose %v failed: %w", res.Shape(), err)
	}

	ret.groups = ret.groups.Upsert(group)
	return ret, nil
}

// Conv1d is a 1D convlution. It relies on Conv2D
func Conv1d(in, filter *Node, kernel, pad, stride, dilation int) (*Node, error) {
	return Conv2d(in, filter, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride}, []int{1, dilation})
}

// MaxPool2D applies the kernel filter to the input node.
// The pad slice can have two different lengths.
//
// - if len(pad) == 2, padding is assume to be symetric, and a padding is adding up *and* down to each dimension
//   paddedOutputH = pad[0] + inputH + pad[0]
//   paddedOutputW = pad[1] + inputW + pad[1]
//
// - if len(pad) == 4, padding is explicit and can be asymmetric.
//   paddedOutputH = pad[0] + inputH + pad[1]
//   paddedOutputW = pad[2] + inputW + pad[3]
func MaxPool2D(x *Node, kernel tensor.Shape, pad, stride []int) (*Node, error) {
	group := encoding.NewGroup("Maxpool")
	xShape := x.Shape()

	// check shape
	if xShape.Dims() != 4 {
		return nil, errors.Errorf("Expected input to have a shape with dimension 4")
	}
	if kernel.Dims() != 2 {
		return nil, errors.Errorf("Expected kernel to have a shape of dimension 2")
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

	h, w := xShape[2], xShape[3]
	kh, kw := kernel[0], kernel[1]

	padNorth := pad[0]
	padWest := pad[1]
	padSouth := pad[0]
	padEast := pad[1]
	if len(pad) == 4 {
		padNorth = pad[0]
		padSouth = pad[1]
		padWest = pad[2]
		padEast = pad[3]
	}

	if h-kh+padNorth+padSouth < 0 {
		// error
		return nil, errors.New("Impossible height/kernel/pad combination")
	}

	if w-kw+padWest+padEast < 0 {
		// error
		return nil, errors.New("Impossible width/kernel/pad combination")
	}

	op := newMaxPoolOp(xShape, kernel, pad, stride)
	retVal, err := ApplyOp(op, x)
	retVal.groups = retVal.groups.Upsert(group)
	return retVal, err
}

// MaxPool1D applies a maxpool on the node x.
func MaxPool1D(x *Node, kernel, pad, stride int) (*Node, error) {
	return MaxPool2D(x, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride})
}

// BatchNorm applies a batchnormalization. This operator can be used in forward pass or for training.
// In an evaluation only, the "op" output can be discared.
// In training phase, γ, β can be discarded and the op should be used.
// Input must be a matrix with shape (B, N) or a 4d tensor with shape (B, C, W, H)
func BatchNorm(x, scale, bias *Node, momentum, epsilon float64) (retVal, γ, β *Node, op *BatchNormOp, err error) {
	dt, err := dtypeOf(x.Type())
	if err != nil {
		return nil, nil, nil, nil, err
	}

	channels := x.Shape()[1]

	mean := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	variance := tensor.New(tensor.Of(dt), tensor.WithShape(channels))

	saveMean := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	saveVar := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	alpha := tensor.New(tensor.Of(dt), tensor.WithShape(channels))
	beta := tensor.New(tensor.Of(dt), tensor.WithShape(channels))

	g := x.Graph()
	dims := x.Shape().Dims()

	if scale == nil {
		scale = NewTensor(g, dt, dims, WithShape(x.Shape().Clone()...), WithName(x.Name()+"_γ"), WithInit(GlorotN(1.0)))
	}
	if bias == nil {
		bias = NewTensor(g, dt, dims, WithShape(x.Shape().Clone()...), WithName(x.Name()+"_β"), WithInit(GlorotN(1.0)))
	}

	op = &BatchNormOp{
		momentum: momentum,
		epsilon:  epsilon,

		runningMean:     mean,
		runningVariance: variance,

		saveMean:     saveMean,
		saveVariance: saveVar,

		alpha: alpha,
		beta:  beta,

		training: true,
		dims:     x.Dims(),
	}

	if retVal, err = ApplyOp(op, x); err != nil {
		return nil, nil, nil, nil, err
	}

	retVal, err = Auto(BroadcastHadamardProd, scale, retVal)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	retVal, err = Auto(BroadcastAdd, retVal, bias)

	return retVal, scale, bias, op, err
}

// GlobalAveragePool2D consumes an input tensor X and applies average pooling across the values in the same channel.
// The expected input shape is BCHW where B is the batch size, C is the number of channels, and H and W are the height and the width of the data.
func GlobalAveragePool2D(x *Node) (*Node, error) {
	return ApplyOp(&globalAveragePoolOp{}, x)
}

func Embedding(weight *Node, indices *Node) (*Node, error) {
	if weight.Shape().Dims() != 2 {
		return nil, errors.Errorf("expected weight to have a shape with dimension 2")
	}

	return ApplyOp(newEmbeddingOp(weight.t, indices.t), weight, indices)
}
