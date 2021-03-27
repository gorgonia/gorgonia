package gorgonia

import (
	"fmt"
	"hash"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/encoding"
	"gorgonia.org/tensor"
)

// AveragePool1D applies a average pool on the node x.
func AveragePool1D(x *Node, kernel, pad, stride int) (*Node, error) {
	return MaxPool2D(x, tensor.Shape{1, kernel}, []int{0, pad}, []int{1, stride})
}

// AveragePool2D applies the average operation to the given input
func AveragePool2D(x *Node, kernel tensor.Shape, pad, stride []int) (*Node, error) {
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

	op := newAvgPoolOp(xShape, kernel, pad, stride)
	retVal, err := ApplyOp(op, x)
	retVal.groups = retVal.groups.Upsert(group)

	return retVal, err
}

type avgPoolOp struct {
	// Shape of Input
	unpaddedB int
	unpaddedC int
	unpaddedH int
	unpaddedW int

	h, w              int // patch height and width
	padNorth, padWest int
	padSouth, padEast int
	explicitPadding   bool
	strideH, strideW  int

	// execution state
	// the mask is only filled at execution time
	mask tensor.Tensor
}

func newAvgPoolOp(inputShape, kernel tensor.Shape, pad, stride []int) *avgPoolOp {
	padNorth := pad[0]
	padWest := pad[1]
	padSouth := pad[0]
	padEast := pad[1]
	explicitPadding := false

	if len(pad) == 4 {
		explicitPadding = true
		padNorth = pad[0]
		padSouth = pad[1]
		padWest = pad[2]
		padEast = pad[3]
	}

	avgPoolOp := &avgPoolOp{
		unpaddedB: inputShape[0],
		unpaddedC: inputShape[1],
		unpaddedH: inputShape[2],
		unpaddedW: inputShape[3],

		h:               kernel[0],
		w:               kernel[1],
		padNorth:        padNorth,
		padWest:         padWest,
		padSouth:        padSouth,
		padEast:         padEast,
		explicitPadding: explicitPadding,
		strideH:         stride[0],
		strideW:         stride[1],
	}

	avgPoolOp.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(avgPoolOp.calcShape(inputShape)...))

	return avgPoolOp
}

func (op *avgPoolOp) Arity() int { return 1 }

// avgPoolOp has this type:
// 		op :: (...) → (...)
func (op *avgPoolOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)
}
func (op *avgPoolOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if s, ok := inputs[0].(tensor.Shape); ok {
		return op.calcShape(s), nil
	}
	return nil, errors.Errorf("Expected a shape")
}

func (op *avgPoolOp) Do(inputs ...Value) (retVal Value, err error) {
	var in, out tensor.Tensor
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}
	inShp := in.Shape()
	out = tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(op.calcShape(inShp)...), tensor.WithEngine(in.Engine()))
	op.do(out, in)
	return out, nil
}

func (op *avgPoolOp) ReturnsPtr() bool     { return false }
func (op *avgPoolOp) CallsExtern() bool    { return false }
func (op *avgPoolOp) OverwritesInput() int { return -1 }
func (op *avgPoolOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "AvgPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *avgPoolOp) Hashcode() uint32 { return simpleHash(op) }

func (op *avgPoolOp) String() string {
	return fmt.Sprintf("AvgPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *avgPoolOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	var in tensor.Tensor
	var err error
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}

	if p, ok := prealloc.(tensor.Tensor); ok {
		op.do(p, in)
		return p, nil
	}
	return nil, errors.Errorf("Expected prealloc to be a tensor")
}

func (op *avgPoolOp) DiffWRT(inputs int) []bool { return []bool{true} }

func (op *avgPoolOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]

	var op2 avgPoolOp
	op2 = *op
	diff := &avgPoolDiffOp{op2}

	var ret *Node
	if ret, err = ApplyOp(diff, input, output, grad); err != nil {
		return nil, err
	}
	return Nodes{ret}, nil
}

func (op *avgPoolOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]
	inputDV, outDV := getDV(input, output)

	var op2 avgPoolOp
	op2 = *op
	diff := &avgPoolDiffOp{op2}

	if _, err = diff.UsePreallocDo(inputDV.d, inputDV.Value, outDV.Value, outDV.d); err != nil {
		return errors.Wrapf(err, doFail, diff)
	}
	return
}

func (op *avgPoolOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	var in tensor.Tensor
	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		return nil, errors.Errorf("Expected input to be a tensor")
	}

	if in.Shape().Dims() != 4 {
		return nil, errors.Errorf("Expected input to have 4 dimensions")
	}
	return in, nil
}

// calcShape calculates the output shape given an input shape
func (op *avgPoolOp) calcShape(s tensor.Shape) tensor.Shape {
	b, c, h, w := s[0], s[1], s[2], s[3]

	pooledH := (h+op.padSouth+op.padNorth-(op.h-1)-1)/op.strideH + 1
	pooledW := (w+op.padEast+op.padWest-(op.w-1)-1)/op.strideW + 1
	return tensor.Shape{b, c, pooledH, pooledW}
}

// do prepares the data, and then dispatches it to the correct (computation) kernel.
// out is the preallocated tensor
func (op *avgPoolOp) do(out, in tensor.Tensor) {
	outShape := out.Shape()
	outStride := out.Strides()[1]
	inShape := in.Shape()
	inStride := in.Strides()[1]
	maskStride := op.mask.Strides()[1]

	b, c, h, w := outShape[0], outShape[1], outShape[2], outShape[3]
	inH, inW := inShape[2], inShape[3]

	if op.mask == nil {
		op.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(op.calcShape(inShape)...))
	}

	maskData := op.mask.Data().([]int)

	switch in.Dtype() {
	case tensor.Float64:
		op.f64s(b, c, h, w, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float64), in.Data().([]float64),
			maskData)
	case tensor.Float32:
		op.f32s(b, c, h, w, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float32), in.Data().([]float32),
			maskData)
	}
}

func (op *avgPoolOp) f32s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float32,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = 0
		maskData[i] = -1
	}

	padH := op.padNorth
	padW := op.padWest
	if op.explicitPadding {
		padH = op.padSouth
		padW = op.padEast
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < outH; ph++ {
				for pw := 0; pw < outW; pw++ {
					hStart := ph*op.strideH - padH
					wStart := pw*op.strideW - padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*outW + pw

					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi
							outData[poolIndex] += inData[i]
							maskData[poolIndex] = i
						}
					}

					outData[poolIndex] /= float32(inW)
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}

func (op *avgPoolOp) f64s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float64,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = 0
		maskData[i] = -1
	}

	padH := op.padNorth
	padW := op.padWest
	if op.explicitPadding {
		padH = op.padSouth
		padW = op.padEast
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < outH; ph++ {
				for pw := 0; pw < outW; pw++ {
					hStart := ph*op.strideH - padH
					wStart := pw*op.strideW - padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*outW + pw

					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi

							outData[poolIndex] += inData[i]
							maskData[poolIndex] = i
						}
					}

					outData[poolIndex] /= float64(inW)
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}

type avgPoolDiffOp struct {
	avgPoolOp
}

func (op *avgPoolDiffOp) Arity() int { return 3 }
func (op *avgPoolDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t, t, t)
}

func (op *avgPoolDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}

func (op *avgPoolDiffOp) Do(inputs ...Value) (Value, error) {
	var in, out, pooled, pooledGrad tensor.Tensor
	var err error
	if in, pooled, pooledGrad, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}

	// out is the gradient of in
	out = tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(in.Shape().Clone()...), tensor.WithEngine(in.Engine()))
	op.do(out, in, pooled, pooledGrad)

	return out, nil
}

func (op *avgPoolDiffOp) ReturnsPtr() bool     { return true }
func (op *avgPoolDiffOp) CallsExtern() bool    { return false }
func (op *avgPoolDiffOp) OverwritesInput() int { return -1 }
func (op *avgPoolDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "AvgPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *avgPoolDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *avgPoolDiffOp) String() string {
	return fmt.Sprintf("AvgPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *avgPoolDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	var in, pooled, pooledGrad tensor.Tensor
	var err error
	if in, pooled, pooledGrad, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}
	if p, ok := prealloc.(tensor.Tensor); ok {
		op.do(p, in, pooled, pooledGrad)
		return prealloc, nil
	}
	return nil, errors.Errorf("Cannot do with PreallocDo - expected PreAlloc to be tensor")
}

func (op *avgPoolDiffOp) checkInput(inputs ...Value) (in, pooled, pooledGrad tensor.Tensor, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	var ok bool
	if in, ok = inputs[0].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected input to be a tensor")
		return
	}

	if in.Shape().Dims() != 4 {
		err = errors.Errorf("Expected input to have 4 dimensions")
		return
	}

	if pooled, ok = inputs[1].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected pooled to be a tensor")
		return
	}
	if pooledGrad, ok = inputs[2].(tensor.Tensor); !ok {
		err = errors.Errorf("Expected pooledGrad to be a tensor")
		return
	}

	return
}

func (op *avgPoolDiffOp) do(inGrad, in, pooled, pooledGrad tensor.Tensor) {
	inShape := inGrad.Shape()
	pooledShape := pooled.Shape()
	pooledStride := pooled.Strides()[1]
	inStride := in.Strides()[1]
	maskStride := op.mask.Strides()[1]
	maskData := op.mask.Data().([]int)

	b, c, h, w := pooledShape[0], pooledShape[1], pooledShape[2], pooledShape[3]
	inH, inW := inShape[2], inShape[3]

	switch in.Dtype() {
	case tensor.Float32:
		inGradData := inGrad.Data().([]float32)
		pooledGradData := pooledGrad.Data().([]float32)
		op.f32s(b, c, h, w, inH, inW,
			inStride, pooledStride, maskStride,
			inGradData, pooledGradData, maskData)
	case tensor.Float64:
		inGradData := inGrad.Data().([]float64)
		pooledGradData := pooledGrad.Data().([]float64)
		op.f64s(b, c, h, w, inH, inW,
			inStride, pooledStride, maskStride,
			inGradData, pooledGradData, maskData)
	}
}

// in is the "bottom", while out is the "top" (bottom being the unpooled, and top being the pooled)
func (op *avgPoolDiffOp) f32s(batches, channels, pooledH, pooledW, inH, inW int,
	inStride, outStride, maskStride int,
	inDiffData, outDiffData []float32,
	maskData []int) {

	// zero out. let's hope go's optimizer is smart enought
	for i := range inDiffData {
		inDiffData[i] = 0
	}

	padH := op.padNorth
	padW := op.padWest
	if op.explicitPadding {
		padH = op.padSouth
		padW = op.padEast
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < pooledH; ph++ {
				for pw := 0; pw < pooledW; pw++ {
					index := ph*pooledW + pw
					inIndex := maskData[index]

					inDiffData[inIndex] += outDiffData[index]
				}
			}

			for ph := 0; ph < inH; ph++ {
				for pw := 0; pw < inW; pw++ {
					hStart := ph*op.strideH - padH
					wStart := pw*op.strideW - padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*inW + pw
					total := float32(0.0)

					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi

							total += inDiffData[i]
						}
					}

					inDiffData[poolIndex] = total / float32(inW)
				}
			}

			outDiffData = outDiffData[outStride:]
			inDiffData = inDiffData[inStride:]
			maskData = maskData[maskStride:]
		}
	}
}

// in is the "bottom", while out is the "top" (bottom being the unpooled, and top being the pooled)
func (op *avgPoolDiffOp) f64s(batches, channels, pooledH, pooledW, inH, inW int,
	inStride, outStride, maskStride int,
	inDiffData, outDiffData []float64,
	maskData []int) {

	// zero out. let's hope go's optimizer is smart enought
	for i := range inDiffData {
		inDiffData[i] = 0
	}

	padH := op.padNorth
	padW := op.padWest
	if op.explicitPadding {
		padH = op.padSouth
		padW = op.padEast
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < pooledH; ph++ {
				for pw := 0; pw < pooledW; pw++ {
					index := ph*pooledW + pw
					inIndex := maskData[index]

					inDiffData[inIndex] += outDiffData[index]
				}
			}

			for ph := 0; ph < inH; ph++ {
				for pw := 0; pw < inW; pw++ {
					hStart := ph*op.strideH - padH
					wStart := pw*op.strideW - padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*inW + pw
					total := float64(0.0)

					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi

							total += inDiffData[i]
						}
					}

					inDiffData[poolIndex] = total / float64(inW)
				}
			}

			outDiffData = outDiffData[outStride:]
			inDiffData = inDiffData[inStride:]
			maskData = maskData[maskStride:]
		}
	}
}
