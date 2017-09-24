package gorgonia

import (
	"fmt"
	"hash"
	"hash/fnv"
	"time"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/leesper/go_rng"
	"github.com/pkg/errors"
)

var (
	_ Op = im2colOp{}
	_ Op = col2imOp{}
	_ Op = &maxPoolOp{}
	_ Op = &maxPoolDiffOp{}
)

/*
	This file contains all the Ops related to building a neural network.

	Bear in mind that not all things that are related to a neural network are here, as not everything
	are encoded as Ops the way theano does it.

	See also: nn.go for functions that relate to neural networks
*/

type randomness byte

const (
	uniform randomness = iota
	gaussian
	binomial
)

type randomOp struct {
	which randomness
	shape tensor.Shape
	dt    tensor.Dtype

	a, b float64 // when uniform, a,b = low, high; when gaussian, a,b = mean, stdev
}

func makeRandomOp(which randomness, dt tensor.Dtype, a, b float64, shape ...int) randomOp {
	return randomOp{
		which: which,
		shape: tensor.Shape(shape),
		dt:    dt,
		a:     a,
		b:     b,
	}
}

func (op randomOp) Arity() int { return 0 }

// randomOp :: a
// randomOp :: Tensor a
func (op randomOp) Type() hm.Type {
	if op.shape.IsScalar() {
		return op.dt
	}
	tt := newTensorType(op.shape.Dims(), op.dt)
	return tt
}

func (op randomOp) InferShape(...DimSizer) (tensor.Shape, error) { return op.shape, nil }

func (op randomOp) Do(...Value) (retVal Value, err error) {
	if op.shape.IsScalar() {
		var v interface{}
		switch op.dt {
		case Float64:
			switch op.which {
			case uniform:
				rand := rng.NewUniformGenerator(time.Now().UnixNano())
				v = rand.Float64Range(op.a, op.b)
			case gaussian:
				rand := rng.NewGaussianGenerator(time.Now().UnixNano())
				v = rand.Gaussian(op.a, op.b)
			case binomial:
				rand := rng.NewBinomialGenerator(time.Now().UnixNano())
				v = float64(rand.Binomial(int64(op.a), op.b))
			}
		case Float32:
			switch op.which {
			case uniform:
				rand := rng.NewUniformGenerator(time.Now().UnixNano())
				v = rand.Float32Range(float32(op.a), float32(op.b))
			case gaussian:
				rand := rng.NewGaussianGenerator(time.Now().UnixNano())
				v = float32(rand.Gaussian(op.a, op.b))
			case binomial:
				rand := rng.NewBinomialGenerator(time.Now().UnixNano())
				v = float32(rand.Binomial(int64(op.a), op.b))
			}
		default:
			return nil, errors.Errorf(nyiFail, "randomOp.do()", op.dt)
		}

		retVal, _ = anyToScalar(v)
		return
	}

	switch op.dt {
	case Float64:
		switch op.which {
		case uniform:
			backing := Uniform64(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		case gaussian:
			backing := Gaussian64(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		case binomial:
			backing := Binomial64(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		}
		return
	case Float32:
		switch op.which {
		case uniform:
			backing := Uniform32(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		case gaussian:
			backing := Gaussian32(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		case binomial:
			backing := Binomial32(op.a, op.b, op.shape...)
			retVal = tensor.New(tensor.WithBacking(backing), tensor.WithShape(op.shape...))
		}
		return
	default:
		return nil, errors.Errorf(nyiFail, "randomOp.do() for non-scalar", op.dt)
	}
}

func (op randomOp) ReturnsPtr() bool     { return false }
func (op randomOp) CallsExtern() bool    { return false }
func (op randomOp) OverwritesInput() int { return -1 }
func (op randomOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "%d%v%f%f", op.which, op.shape, op.a, op.b)
}

func (op randomOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op randomOp) String() string {
	return fmt.Sprintf("%v(%v, %v) - %v", op.which, op.a, op.b, op.shape)
}

type im2colOp struct {
	h, w             int // kernel height and width
	padH, padW       int
	strideH, strideW int
}

func (op im2colOp) Arity() int { return 1 }

// im2col :: (Floats a) ⇒ a →  a
func (op im2colOp) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op im2colOp) InferShape(shapes ...DimSizer) (retVal tensor.Shape, err error) {
	if err = checkArity(op, len(shapes)); err != nil {
		return
	}

	if s, ok := shapes[0].(tensor.Shape); ok {
		return op.calcShape(s), nil
	}
	return nil, errors.Errorf("expected tensor.Shape. got %T instead", shapes[0])
}

func (op im2colOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	im := inputs[0]

	// todo type check values
	// todo shape check values

	retShape := op.calcShape(im.Shape())
	prealloc := tensor.New(tensor.Of(im.Dtype()), tensor.WithShape(retShape...))

	return op.do(prealloc, im)
}

func (op im2colOp) ReturnsPtr() bool     { return false }
func (op im2colOp) CallsExtern() bool    { return false }
func (op im2colOp) OverwritesInput() int { return -1 }

func (op im2colOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "im2col:%d-%d-%d-%d-%d-%d", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op im2colOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op im2colOp) String() string {
	return fmt.Sprintf("im2col<(%d,%d), (%d, %d), (%d,%d)>", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op im2colOp) DiffWRT(i int) []bool { return []bool{true} }

func (op im2colOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	im := inputs[0]
	s := im.Shape()
	if s.TotalSize() != 4 {
		return nil, errors.Errorf("Expected input to have a shape with 4 dims")
	}
	var unpaddedB, unpaddedC, unpaddedH, unpaddedW int
	unpaddedB, unpaddedC, unpaddedH, unpaddedW = s[0], s[1], s[2], s[3]
	diffOp := col2imOp{
		unpaddedB: unpaddedB,
		unpaddedC: unpaddedC,
		unpaddedH: unpaddedH,
		unpaddedW: unpaddedW,

		h:       op.h,
		w:       op.w,
		padH:    op.padH,
		padW:    op.padW,
		strideH: op.strideH,
		strideW: op.strideW,
	}

	var ret *Node
	if ret, err = applyOp(diffOp, grad); err != nil {
		return
	}
	retVal = Nodes{ret}
	return
}

func (op im2colOp) calcShape(s tensor.Shape) (retVal tensor.Shape) {
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	h2 := (h+2*op.padH-op.h)/op.strideH + 1
	w2 := (w+2*op.padW-op.w)/op.strideW + 1
	retVal = tensor.Shape(tensor.BorrowInts(4))

	// todo: double check this with tests
	retVal[0] = b
	retVal[1] = h2
	retVal[2] = w2
	retVal[3] = c * op.w * op.h

	return
}

func (op im2colOp) do(prealloc, input Value) (retVal Value, err error) {
	// extract bchw - this bit can be expanded in the future, but for now we only support bchw
	s := input.Shape()
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	switch input.Dtype() {
	case tensor.Float64:
		for i := 0; i < b; i++ {
			op.f64s(c, h, w, input.Data().([]float64), prealloc.Data().([]float64))
		}
	case tensor.Float32:
		for i := 0; i < b; i++ {
			op.f32s(c, h, w, input.Data().([]float32), prealloc.Data().([]float32))
		}
	default:
		return nil, errors.Errorf(nyiFail, "im2col", input.Dtype())
	}
	return prealloc, nil
}

func (op im2colOp) f64s(channels, height, width int, im, col []float64) {
	retHeight := (height+2*op.padH-op.h)/op.strideH + 1
	retWidth := (width+2*op.padW-op.w)/op.strideW + 1
	retChans := channels * op.h * op.w

	for c := 0; c < retChans; c++ {
		widthOffset := c % op.w
		heightOffset := (c / op.w) % op.h
		imChan := c / op.h / op.w

		for h := 0; h < retHeight; h++ {
			for w := 0; w < retWidth; w++ {
				padH := h*op.strideH - op.padH + heightOffset
				padW := w*op.strideW - op.padW + widthOffset

				idx := retChans*retWidth*h + retChans*w + c
				if padH >= 0 && padH < height && padW >= 0 && padW < width {
					imIdx := (imChan*height+padH)*width + padW
					col[idx] = im[imIdx]
				} else {
					col[idx] = 0
				}
			}
		}
	}
}

func (op im2colOp) f32s(channels, height, width int, im, col []float32) {
	retHeight := (height+2*op.padH-op.h)/op.strideH + 1
	retWidth := (width+2*op.padW-op.w)/op.strideW + 1
	retChans := channels * op.h * op.w

	for c := 0; c < retChans; c++ {
		widthOffset := c % op.w
		heightOffset := (c / op.w) % op.h
		imChan := c / op.h / op.w

		for h := 0; h < retHeight; h++ {
			for w := 0; w < retWidth; w++ {
				padH := h*op.strideH - op.padH + heightOffset
				padW := w*op.strideW - op.padW + widthOffset

				idx := retChans*retWidth*h + retChans*w + c
				if padH >= 0 && padH < height && padW >= 0 && padW < width {
					imIdx := (imChan*height+padH)*width + padW
					col[idx] = im[imIdx]
				} else {
					col[idx] = 0
				}
			}
		}
	}
}

/*
// Experimental fast(er) version... which may be premature optimization, so it's actually commented out
// and it's not actually fully implemented. I'm quite sure there are some holes in my logics

func (op im2colOp) f64s(channels, height, width int, im, col []float64){
	for c := range col {
		padH := c / (outHeight * outWidth)
		padW := c % (outHeight * outWidth)

		i := (padH /  op.h) / op.h
		j := (padW / outWidth) + (padH / op.h) % op.w
		k := padW % outWide + padH / op.h

		idx :=
		imIdx :=
		if (j >= op.padH  &&j < op.padH+inHeight) && (k >= op.padW && k < op.padW + inWidth) {
			col[idx] = im[imIdx]
		} else {
			col[idx]=0
		}
	}
}

*/

type col2imOp struct {
	// input shapes of im2col
	unpaddedB int
	unpaddedC int
	unpaddedH int
	unpaddedW int

	h, w             int // patch height and width
	padH, padW       int
	strideH, strideW int
}

func (op col2imOp) Arity() int { return 1 }

// im2col :: (Floats a) ⇒ a →  a
func (op col2imOp) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op col2imOp) InferShape(shapes ...DimSizer) (retVal tensor.Shape, err error) {
	return tensor.Shape{op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW}, nil
}

func (op col2imOp) Do(inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	im := inputs[0]

	// todo type check values
	// todo shape check values

	retShape := tensor.Shape{op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW}
	prealloc := tensor.New(tensor.Of(im.Dtype()), tensor.WithShape(retShape...))

	return op.do(prealloc, im)
}

func (op col2imOp) ReturnsPtr() bool     { return false }
func (op col2imOp) CallsExtern() bool    { return false }
func (op col2imOp) OverwritesInput() int { return -1 }

func (op col2imOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "col2im:%d-%d-%d-%d-%d-%d", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op col2imOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op col2imOp) String() string {
	return fmt.Sprintf("col2im<(%d,%d), (%d, %d), (%d,%d)>", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op col2imOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}
	return op.do(prealloc, inputs[0])
}

func (op col2imOp) do(prealloc, input Value) (retVal Value, err error) {
	b := op.unpaddedB
	c := op.unpaddedC
	h := op.unpaddedH
	w := op.unpaddedW

	switch input.Dtype() {
	case tensor.Float64:
		for i := 0; i < b; i++ {
			op.f64s(c, h, w, input.Data().([]float64), prealloc.Data().([]float64))
		}
	case tensor.Float32:
		for i := 0; i < b; i++ {
			op.f32s(c, h, w, input.Data().([]float32), prealloc.Data().([]float32))
		}
	default:
		return nil, errors.Errorf(nyiFail, "col2im", input.Dtype())
	}
	return
}

func (op col2imOp) f64s(channels, height, width int, col, im []float64) {
	// memset im to 0
	for i := 0; i < height*width*channels; i++ {
		im[i] = 0
	}

	colHeight := (height+2*op.padH-op.h)/op.strideH + 1
	colWidth := (width+2*op.padW-op.w)/op.strideW + 1
	colChans := channels * op.h * op.w

	for c := 0; c < colChans; c++ {
		widthOffset := c % op.w
		heightOffset := (c / op.w) % op.h
		imChan := c / op.w / op.h
		for h := 0; h < colHeight; h++ {
			for w := 0; w < colWidth; w++ {
				padH := h*op.strideH - op.padH + heightOffset
				padW := w*op.strideW - op.padW + widthOffset
				if padH >= 0 && padH < height && padW > 0 && padW < width {
					imIdx := (imChan*height+padH)*width + padW
					colIdx := colChans*colWidth*h + colChans*w + c
					im[imIdx] += col[colIdx]
				}
			}
		}
	}
}

func (op col2imOp) f32s(channels, height, width int, col, im []float32) {
	// memset im to 0
	for i := 0; i < height*width*channels; i++ {
		im[i] = 0
	}

	colHeight := (height+2*op.padH-op.h)/op.strideH + 1
	colWidth := (width+2*op.padW-op.w)/op.strideW + 1
	colChans := channels * op.h * op.w

	for c := 0; c < colChans; c++ {
		widthOffset := c % op.w
		heightOffset := (c / op.w) % op.h
		imChan := c / op.w / op.h
		for h := 0; h < colHeight; h++ {
			for w := 0; w < colWidth; w++ {
				padH := h*op.strideH - op.padH + heightOffset
				padW := w*op.strideW - op.padW + widthOffset
				if padH >= 0 && padH < height && padW > 0 && padW < width {
					imIdx := (imChan*height+padH)*width + padW
					colIdx := colChans*colWidth*h + colChans*w + c
					im[imIdx] += col[colIdx]
				}
			}
		}
	}
}

// It's important to note that this op actually produces TWO values - one argmax, which will be used
// as a mask, and the actual pooled value.
//
// The argmax is stored as an internal state and is not exposed to anything outside the op.
// There are alternative ways of designing this op, but they all don't particularly seem nice.
// Caffe's technique seemed the nicest.
type maxPoolOp struct {
	// Shape of Input
	unpaddedB int
	unpaddedC int
	unpaddedH int
	unpaddedW int

	h, w             int // patch height and width
	padH, padW       int
	strideH, strideW int

	// execution state
	// the mask is only filled at execution time
	mask tensor.Tensor
}

func (op *maxPoolOp) Arity() int { return 1 }

// maxPoolOp has this type:
// 		op :: (...) → (...)
func (op *maxPoolOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)
}
func (op *maxPoolOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	if s, ok := inputs[0].(tensor.Shape); ok {
		return op.calcShape(s), nil
	}
	return nil, errors.Errorf("Expected a shape")
}

func (op *maxPoolOp) Do(inputs ...Value) (retVal Value, err error) {
	var in, out tensor.Tensor
	if in, err = op.checkInput(inputs...); err != nil {
		return nil, err
	}

	inShp := in.Shape()

	out = tensor.New(tensor.Of(in.Dtype()), tensor.WithShape(op.calcShape(inShp)...), tensor.WithEngine(in.Engine()))
	op.do(out, in)
	return out, nil
}

func (op *maxPoolOp) ReturnsPtr() bool     { return true }
func (op *maxPoolOp) CallsExtern() bool    { return false }
func (op *maxPoolOp) OverwritesInput() int { return -1 }
func (op *maxPoolOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "MaxPool{%d, %d, %d, %d}(%d, %d %d, %d, %d %d)",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op *maxPoolOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *maxPoolOp) String() string {
	return fmt.Sprintf("MaxPool{%d, %d, %d, %d}(%d, %d %d, %d, %d %d)",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op *maxPoolOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
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

func (op *maxPoolOp) DiffWRT(inputs int) []bool { return []bool{true} }

func (op *maxPoolOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	input := inputs[0]

	var op2 maxPoolOp
	op2 = *op
	diff := &maxPoolDiffOp{op2}

	var ret *Node
	if ret, err = applyOp(diff, input, output, grad); err != nil {
		return nil, err
	}
	return Nodes{ret}, nil
}

func (op *maxPoolOp) checkInput(inputs ...Value) (tensor.Tensor, error) {
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
func (op *maxPoolOp) calcShape(s tensor.Shape) tensor.Shape {
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	pooledH := ceilDivInt((h + 2*op.padH - op.h), op.strideH)
	pooledW := ceilDivInt((w + 2*op.padW - op.w), op.strideW)
	return tensor.Shape{b, c, pooledH, pooledW}
}

// do prepares the data, and then dispatches it to the correct (computation) kernel.
// out is the preallocated tensor
func (op *maxPoolOp) do(out, in tensor.Tensor) {
	outShape := out.Shape()
	outStride := out.Strides()[1]
	inShape := in.Shape()
	inStride := in.Strides()[1]
	maskStride := op.mask.Strides()[1]

	batches := outShape[0]
	channels := outShape[1]
	outH := outShape[2]
	outW := outShape[3]

	inH := inShape[2]
	inW := inShape[3]

	if op.mask == nil {
		op.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(op.calcShape(inShape)...))
	}

	maskData := op.mask.Data().([]int)

	switch in.Dtype() {
	case tensor.Float64:
		op.f64s(batches, channels, outH, outW, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float64), in.Data().([]float64),
			maskData)
	case tensor.Float32:
		op.f32s(batches, channels, outH, outW, inH, inW,
			outStride, inStride, maskStride,
			out.Data().([]float32), in.Data().([]float32),
			maskData)
	}
}

func (op *maxPoolOp) f32s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float32,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = -maxFloat32
		maskData[i] = -1
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < outH; ph++ {
				for pw := 0; pw < outW; pw++ {
					hStart := ph*op.strideH - op.padH
					wStart := pw*op.strideW - op.padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*outW + pw
					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi
							if inData[i] > outData[poolIndex] {
								outData[poolIndex] = inData[i]
								maskData[poolIndex] = i
							}
						}
					}
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}

func (op *maxPoolOp) f64s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float64,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = -maxFloat64
		maskData[i] = -1
	}

	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < outH; ph++ {
				for pw := 0; pw < outW; pw++ {
					hStart := ph*op.strideH - op.padH
					wStart := pw*op.strideW - op.padW
					hEnd := minInt(hStart+op.h, inH)
					wEnd := minInt(wStart+op.w, inW)
					hStart = maxInt(hStart, 0)
					wStart = maxInt(wStart, 0)

					poolIndex := ph*outW + pw

					for hi := hStart; hi < hEnd; hi++ {
						for wi := wStart; wi < wEnd; wi++ {
							i := hi*inW + wi
							if inData[i] > outData[poolIndex] {
								outData[poolIndex] = inData[i]
								maskData[poolIndex] = i
							}
						}
					}
				}
			}
			// skip by strides
			inData = inData[inStride:]
			outData = outData[outStride:]
			maskData = maskData[maskStride:]
		}
	}
}

type maxPoolDiffOp struct {
	maxPoolOp
}

func (op *maxPoolDiffOp) Arity() int { return 3 }
func (op *maxPoolDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t, t, t)
}

func (op *maxPoolDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()
	return s, nil
}

func (op *maxPoolDiffOp) Do(inputs ...Value) (Value, error) {
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
func (op *maxPoolDiffOp) ReturnsPtr() bool     { return true }
func (op *maxPoolDiffOp) CallsExtern() bool    { return false }
func (op *maxPoolDiffOp) OverwritesInput() int { return -1 }
func (op *maxPoolDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "MaxPoolDiff{%d, %d, %d, %d}(%d, %d %d, %d, %d %d)",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op *maxPoolDiffOp) Hashcode() uint32 {
	h := fnv.New32a()
	op.WriteHash(h)
	return h.Sum32()
}

func (op *maxPoolDiffOp) String() string {
	return fmt.Sprintf("MaxPoolDiff{%d, %d, %d, %d}(%d, %d %d, %d, %d %d)",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padH, op.padW, op.strideH, op.strideW)
}

func (op *maxPoolDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
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

func (op *maxPoolDiffOp) checkInput(inputs ...Value) (in, pooled, pooledGrad tensor.Tensor, err error) {
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

func (op *maxPoolDiffOp) do(inGrad, in, pooled, pooledGrad tensor.Tensor) {
	pooledShape := pooled.Shape()
	pooledStride := pooled.Strides()[1]
	inStride := in.Strides()[1]
	maskStride := op.mask.Strides()[1]

	batches := pooledShape[0]
	channels := pooledShape[1]
	height := pooledShape[2]
	width := pooledShape[3]

	maskData := op.mask.Data().([]int)

	switch in.Dtype() {
	case tensor.Float32:
		inGradData := inGrad.Data().([]float32)
		pooledGradData := pooledGrad.Data().([]float32)
		op.f32s(batches, channels, height, width,
			inStride, pooledStride, maskStride,
			inGradData, pooledGradData, maskData)
	case tensor.Float64:
		inGradData := inGrad.Data().([]float64)
		pooledGradData := pooledGrad.Data().([]float64)
		op.f64s(batches, channels, height, width,
			inStride, pooledStride, maskStride,
			inGradData, pooledGradData, maskData)
	}
}

// in is the "bottom", while out is the "top" (bottom being the unpooled, and top being the pooled)
func (op *maxPoolDiffOp) f32s(batches, channels, pooledH, pooledW int,
	inStride, outStride, maskStride int,
	inDiffData, outDiffData []float32,
	maskData []int) {

	// zero out. let's hope go's optimizer is smart enought
	for i := range inDiffData {
		inDiffData[i] = 0
	}

	// this loop can be goroutine'd
	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < pooledH; ph++ {
				for pw := 0; pw < pooledW; pw++ {
					index := ph*pooledW + pw
					inIndex := maskData[index]
					inDiffData[inIndex] += outDiffData[index]
				}
			}
			outDiffData = outDiffData[outStride:]
			inDiffData = inDiffData[inStride:]
			maskData = maskData[maskStride:]
		}
	}
}

// in is the "bottom", while out is the "top" (bottom being the unpooled, and top being the pooled)
func (op *maxPoolDiffOp) f64s(batches, channels, pooledH, pooledW int,
	inStride, outStride, maskStride int,
	inDiffData, outDiffData []float64,
	maskData []int) {

	// zero out. let's hope go's optimizer is smart enought
	for i := range inDiffData {
		inDiffData[i] = 0
	}

	// this loop can be goroutine'd
	for b := 0; b < batches; b++ {
		for c := 0; c < channels; c++ {
			for ph := 0; ph < pooledH; ph++ {
				for pw := 0; pw < pooledW; pw++ {
					index := ph*pooledW + pw
					inIndex := maskData[index]
					inDiffData[inIndex] += outDiffData[index]
				}
			}
			outDiffData = outDiffData[outStride:]
			inDiffData = inDiffData[inStride:]
			maskData = maskData[maskStride:]
		}
	}
}
