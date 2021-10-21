package gorgonia

import (
	"fmt"
	"hash"
	"math"
	"time"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	rng "github.com/leesper/go_rng"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// Sanity checks
var (
	_ SDOp = im2colOp{}
	_ Op   = col2imOp{}
	_ Op   = &maxPoolOp{}
	_ Op   = &maxPoolDiffOp{}
	_ Op   = &BatchNormOp{}
	_ Op   = &batchnormDiffOp{}
	_ Op   = &globalAveragePoolOp{}
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

func (op randomOp) Hashcode() uint32 { return simpleHash(op) }

func (op randomOp) String() string {
	return fmt.Sprintf("%v(%v, %v) - %v", op.which, op.a, op.b, op.shape)
}

type im2colOp struct {
	h, w                 int // kernel height and width
	padH, padW           int
	strideH, strideW     int
	dilationH, dilationW int
}

func makeIm2ColOp(kernelHeight, kernelWidth, padHeight, padWidth, strideHeight, strideWidth, dilationHeight, dilationWidth int) im2colOp {
	return im2colOp{
		h:         kernelHeight,
		w:         kernelWidth,
		padH:      padHeight,
		padW:      padWidth,
		strideH:   strideHeight,
		strideW:   strideWidth,
		dilationH: dilationHeight,
		dilationW: dilationWidth,
	}
}

func (op im2colOp) Arity() int { return 1 }

// im2col :: (Floats a) ⇒ Tensor a →  Tensor a
func (op im2colOp) Type() hm.Type {
	t := makeTensorType(4, hm.TypeVariable('a'))
	return hm.NewFnType(t, t)
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

func (op im2colOp) Hashcode() uint32 { return simpleHash(op) }

func (op im2colOp) String() string {
	return fmt.Sprintf("im2col<(%d,%d), (%d, %d), (%d,%d) (%d, %d)>", op.h, op.w, op.padH, op.padW, op.strideH, op.strideW, op.dilationH, op.dilationW)
}

func (op im2colOp) DiffWRT(i int) []bool { return []bool{true} }

func (op im2colOp) SymDiff(inputs Nodes, output, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	im := inputs[0]
	s := im.Shape()
	if s.Dims() != 4 {
		return nil, errors.Errorf("Expected input to have a shape with 4 dims")
	}
	var unpaddedB, unpaddedC, unpaddedH, unpaddedW int
	unpaddedB, unpaddedC, unpaddedH, unpaddedW = s[0], s[1], s[2], s[3]
	diffOp := col2imOp{
		unpaddedB: unpaddedB,
		unpaddedC: unpaddedC,
		unpaddedH: unpaddedH,
		unpaddedW: unpaddedW,

		im2colOp: op,
	}

	var ret *Node
	if ret, err = ApplyOp(diffOp, grad); err != nil {
		return
	}
	retVal = Nodes{ret}
	return
}

func (op im2colOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	im := inputs[0]
	s := im.Shape()
	imv, colv := getDV(im, output)

	var unpaddedB, unpaddedC, unpaddedH, unpaddedW int
	unpaddedB, unpaddedC, unpaddedH, unpaddedW = s[0], s[1], s[2], s[3]
	diffOp := col2imOp{
		unpaddedB: unpaddedB,
		unpaddedC: unpaddedC,
		unpaddedH: unpaddedH,
		unpaddedW: unpaddedW,

		im2colOp: op,
	}

	if _, err = diffOp.UsePreallocDo(imv.d, colv.d); err != nil {
		return errors.Wrapf(err, doFail, diffOp)
	}
	return
}

func (op im2colOp) calcShape(s tensor.Shape) (retVal tensor.Shape) {
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	retHeight, retWidth := op.retHW(h, w)
	retVal = tensor.Shape(tensor.BorrowInts(4))

	// todo: double check this with tests
	retVal[0] = b
	retVal[1] = retHeight
	retVal[2] = retWidth
	retVal[3] = c * op.w * op.h

	return
}

func (op im2colOp) retHW(h, w int) (retHeight, retWidth int) {
	retHeight = (h+2*op.padH-(op.dilationH*(op.h-1)+1))/op.strideH + 1
	retWidth = (w+2*op.padW-(op.dilationW*(op.w-1)+1))/op.strideW + 1
	return
}

func (op im2colOp) do(prealloc, input Value) (retVal Value, err error) {
	inputT := input.(*tensor.Dense)
	outputT := prealloc.(*tensor.Dense)

	// extract bchw - this bit can be expanded in the future, but for now we only support bchw
	s := inputT.Shape()
	b := s[0]
	c := s[1]
	h := s[2]
	w := s[3]

	inputStrides := inputT.Strides()
	retHeight, retWidth := op.retHW(h, w)
	batchStrideIm := inputStrides[0]
	batchStrideCol := outputT.Strides()[0]
	chanStride := h * w
	inRowStride := inputStrides[2]

	switch input.Dtype() {
	case tensor.Float64:
		imData := input.Data().([]float64)
		colData := prealloc.Data().([]float64)
		for i := 0; i < b; i++ {
			imStart := i * batchStrideIm
			colStart := i * batchStrideCol
			imEnd := imStart + batchStrideIm
			colEnd := colStart + batchStrideCol

			if imEnd >= len(imData) {
				imEnd = len(imData)
			}
			if colEnd >= len(colData) {
				colEnd = len(colData)
			}

			op.f64s(c, h, w, chanStride, inRowStride, retHeight, retWidth, imData[imStart:imEnd], colData[colStart:colEnd])
		}
	case tensor.Float32:
		imData := input.Data().([]float32)
		colData := prealloc.Data().([]float32)
		for i := 0; i < b; i++ {
			imStart := i * batchStrideIm
			colStart := i * batchStrideCol
			imEnd := imStart + batchStrideIm
			colEnd := colStart + batchStrideCol

			if imEnd >= len(imData) {
				imEnd = len(imData)
			}
			if colEnd >= len(colData) {
				colEnd = len(colData)
			}

			op.f32s(c, h, w, chanStride, inRowStride, retHeight, retWidth, imData[imStart:imEnd], colData[colStart:colEnd])
		}
	default:
		return nil, errors.Errorf(nyiFail, "im2col", input.Dtype())
	}
	return prealloc, nil
}

func (op im2colOp) f64s(chans, height, width, chanStride, inRowStride, retHeight, retWidth int, im, col []float64) {
	colIdx := 0
	var inputRow int
	var inputCol int
	for outputRow := 0; outputRow < retHeight; outputRow++ {
		for outputCol := 0; outputCol < retWidth; outputCol++ {
			for ch := 0; ch < chans; ch++ {
				for kernelRow := 0; kernelRow < op.h; kernelRow++ {
					inputRow = -op.padH + kernelRow*op.dilationH + outputRow*op.strideH
					for kernelCol := 0; kernelCol < op.w; kernelCol++ {
						if inputRow < 0 || inputRow >= height {
							col[colIdx] = 0
							colIdx++
							continue
						}
						inputCol = -op.padW + kernelCol*op.dilationW + outputCol*op.strideW
						if inputCol < 0 || inputCol >= width {
							col[colIdx] = 0
							colIdx++
						} else {
							imIdx := chanStride*ch + inputRow*width + inputCol
							col[colIdx] = im[imIdx]
							colIdx++
						}
					}
				}
			}
		}
	}
}

func (op im2colOp) f32s(chans, height, width, chanStride, inRowStride, retHeight, retWidth int, im, col []float32) {
	colIdx := 0
	var inputRow int
	var inputCol int
	for outputRow := 0; outputRow < retHeight; outputRow++ {
		for outputCol := 0; outputCol < retWidth; outputCol++ {
			for ch := 0; ch < chans; ch++ {
				for kernelRow := 0; kernelRow < op.h; kernelRow++ {
					inputRow = -op.padH + kernelRow*op.dilationH + outputRow*op.strideH
					for kernelCol := 0; kernelCol < op.w; kernelCol++ {
						if inputRow < 0 || inputRow >= height {
							col[colIdx] = 0
							colIdx++
							continue
						}
						inputCol = -op.padW + kernelCol*op.dilationW + outputCol*op.strideW
						if inputCol < 0 || inputCol >= width {
							col[colIdx] = 0
							colIdx++
						} else {
							imIdx := chanStride*ch + inputRow*width + inputCol
							col[colIdx] = im[imIdx]
							colIdx++
						}
					}
				}
			}
		}
	}
}

type col2imOp struct {
	// input shapes of im2col
	unpaddedB int
	unpaddedC int
	unpaddedH int
	unpaddedW int

	im2colOp
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

func (op col2imOp) Hashcode() uint32 { return simpleHash(op) }

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
	retHeight := op.unpaddedH
	retWidth := op.unpaddedW
	batchStrideIm := c * retHeight * retWidth

	s := input.Shape()
	h := s[1]
	w := s[2]
	chanStride := retHeight * retWidth
	batchStrideCol := h * w * s[3]

	var imStart, imEnd, colStart, colEnd int
	imEnd = imStart + batchStrideIm
	colEnd = colStart + batchStrideCol

	switch input.Dtype() {
	case tensor.Float64:
		colData := input.Data().([]float64)
		imData := prealloc.Data().([]float64)
		for i := 0; i < b; i++ {
			op.f64s(c, retHeight, retWidth, chanStride, h, w, colData[colStart:colEnd], imData[imStart:imEnd])

			colStart += batchStrideCol
			colEnd += batchStrideCol

			imStart += batchStrideIm
			imEnd += batchStrideIm

			if imEnd > len(imData) {
				imEnd = len(imData)
			}
			if colEnd > len(colData) {
				colEnd = len(colData)
			}
		}
	case tensor.Float32:
		colData := input.Data().([]float32)
		imData := prealloc.Data().([]float32)
		for i := 0; i < b; i++ {
			op.f32s(c, retHeight, retWidth, chanStride, h, w, colData[colStart:colEnd], imData[imStart:imEnd])

			colStart += batchStrideCol
			colEnd += batchStrideCol

			imStart += batchStrideIm
			imEnd += batchStrideIm

			if imEnd > len(imData) {
				imEnd = len(imData)
			}
			if colEnd > len(colData) {
				colEnd = len(colData)
			}
		}
	default:
		return nil, errors.Errorf(nyiFail, "col2im", input.Dtype())
	}

	return prealloc, nil
}

func (op col2imOp) f64s(chans, height, width, chanStride, retHeight, retWidth int, col, im []float64) {
	// memset im to 0
	for i := 0; i < len(im); i++ {
		im[i] = 0
	}
	colIdx := 0
	var inputRow int
	var inputCol int
	for outputRow := 0; outputRow < retHeight; outputRow++ {
		for outputCol := 0; outputCol < retWidth; outputCol++ {
			for ch := 0; ch < chans; ch++ {
				for kernelRow := 0; kernelRow < op.h; kernelRow++ {
					inputRow = -op.padH + kernelRow*op.dilationH + outputRow*op.strideH
					for kernelCol := 0; kernelCol < op.w; kernelCol++ {
						if inputRow < 0 || inputRow >= height {
							colIdx++
							continue
						}
						inputCol = -op.padW + kernelCol*op.dilationW + outputCol*op.strideW
						if inputCol >= 0 && inputCol < width {
							imIdx := chanStride*ch + inputRow*width + inputCol
							im[imIdx] += col[colIdx]
						}
						colIdx++
					}
				}
			}
		}
	}
}

func (op col2imOp) f32s(chans, height, width, chanStride, retHeight, retWidth int, col, im []float32) {
	// memset im to 0
	for i := 0; i < len(im); i++ {
		im[i] = 0
	}
	colIdx := 0
	var inputRow int
	var inputCol int
	for outputRow := 0; outputRow < retHeight; outputRow++ {
		for outputCol := 0; outputCol < retWidth; outputCol++ {
			for ch := 0; ch < chans; ch++ {
				for kernelRow := 0; kernelRow < op.h; kernelRow++ {
					inputRow = -op.padH + kernelRow*op.dilationH + outputRow*op.strideH
					for kernelCol := 0; kernelCol < op.w; kernelCol++ {
						if inputRow < 0 || inputRow >= height {
							colIdx++
							continue
						}
						inputCol = -op.padW + kernelCol*op.dilationW + outputCol*op.strideW
						if inputCol >= 0 && inputCol < width {
							imIdx := chanStride*ch + inputRow*width + inputCol
							im[imIdx] += col[colIdx]
						}
						colIdx++
					}
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

	h, w              int // patch height and width
	padNorth, padWest int
	padSouth, padEast int
	explicitPadding   bool
	strideH, strideW  int

	// execution state
	// the mask is only filled at execution time
	mask tensor.Tensor
}

func newMaxPoolOp(inputShape, kernel tensor.Shape, pad, stride []int) *maxPoolOp {
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
	maxpoolOp := &maxPoolOp{
		// Shape of Input
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
	maxpoolOp.mask = tensor.New(tensor.Of(tensor.Int), tensor.WithShape(maxpoolOp.calcShape(inputShape)...))
	return maxpoolOp
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

func (op *maxPoolOp) ReturnsPtr() bool     { return false }
func (op *maxPoolOp) CallsExtern() bool    { return false }
func (op *maxPoolOp) OverwritesInput() int { return -1 }
func (op *maxPoolOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "MaxPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *maxPoolOp) Hashcode() uint32 { return simpleHash(op) }

func (op *maxPoolOp) String() string {
	return fmt.Sprintf("MaxPool{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
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
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]

	var op2 maxPoolOp
	op2 = *op
	diff := &maxPoolDiffOp{op2}

	var ret *Node
	if ret, err = ApplyOp(diff, input, output, grad); err != nil {
		return nil, err
	}
	return Nodes{ret}, nil
}

func (op *maxPoolOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) (err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]
	inputDV, outDV := getDV(input, output)

	var op2 maxPoolOp
	op2 = *op
	diff := &maxPoolDiffOp{op2}

	if _, err = diff.UsePreallocDo(inputDV.d, inputDV.Value, outDV.Value, outDV.d); err != nil {
		return errors.Wrapf(err, doFail, diff)
	}
	return
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
	b, c, h, w := s[0], s[1], s[2], s[3]

	pooledH := (h+op.padSouth+op.padNorth-(op.h-1)-1)/op.strideH + 1
	pooledW := (w+op.padEast+op.padWest-(op.w-1)-1)/op.strideW + 1
	return tensor.Shape{b, c, pooledH, pooledW}
}

func (op *maxPoolOp) strideValue(strides []int) int {
	if len(strides) < 2 {
		return 0
	}

	return strides[1]
}

// do prepares the data, and then dispatches it to the correct (computation) kernel.
// out is the preallocated tensor
func (op *maxPoolOp) do(out, in tensor.Tensor) {
	outShape := out.Shape()
	outStride := op.strideValue(out.Strides())

	inShape := in.Shape()
	inStride := op.strideValue(in.Strides())

	maskStride := op.strideValue(op.mask.Strides())

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

func (op *maxPoolOp) f32s(batches, channels, outH, outW, inH, inW,
	outStride, inStride, maskStride int,
	outData, inData []float32,
	maskData []int) {

	// set values
	for i := range outData {
		outData[i] = -maxFloat32
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
	fmt.Fprintf(h, "MaxPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
}

func (op *maxPoolDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *maxPoolDiffOp) String() string {
	return fmt.Sprintf("MaxPoolDiff{%d, %d, %d, %d}(kernel: (%d, %d), pad: (%d, %d), stride: (%d, %d))",
		op.unpaddedB, op.unpaddedC, op.unpaddedH, op.unpaddedW,
		op.h, op.w, op.padNorth, op.padWest, op.strideH, op.strideW)
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
	pooledStride := op.strideValue(pooled.Strides())
	inStride := in.Strides()[1]
	maskStride := op.strideValue(op.mask.Strides())
	maskData := op.mask.Data().([]int)

	b, c, h, w := pooledShape[0], pooledShape[1], pooledShape[2], pooledShape[3]
	switch in.Dtype() {
	case tensor.Float32:
		inGradData := inGrad.Data().([]float32)
		pooledGradData := pooledGrad.Data().([]float32)
		op.f32s(b, c, h, w,
			inStride, pooledStride, maskStride,
			inGradData, pooledGradData, maskData)
	case tensor.Float64:
		inGradData := inGrad.Data().([]float64)
		pooledGradData := pooledGrad.Data().([]float64)
		op.f64s(b, c, h, w,
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

// clampOp is a constant clamping operation
type clampOp struct {
	min, max Scalar
}

func (op *clampOp) Arity() int { return 1 }

func (op *clampOp) Type() hm.Type {
	return hm.NewFnType(hm.TypeVariable('a'), hm.TypeVariable('a'))
}

func (op *clampOp) InferShape(shps ...DimSizer) (tensor.Shape, error) {
	return shps[0].(tensor.Shape), nil
}

func (op *clampOp) Do(vals ...Value) (Value, error) {
	return nil, nil
}

func (op *clampOp) ReturnsPtr() bool { return true }

func (op *clampOp) CallsExtern() bool { return false }

func (op *clampOp) OverwritesInput() int { return 0 }

func (op *clampOp) WriteHash(h hash.Hash) { fmt.Fprintf(h, "ConstClamp{%f, %f}()", op.min, op.max) }

func (op *clampOp) Hashcode() uint32 { return simpleHash(op) }
func (op *clampOp) String() string   { return fmt.Sprintf("ConstClamp{%f, %f}()", op.min, op.max) }

// BatchNormOp is a batch normalization process as described by Ioffe and Szegedy (2015) -
// http://arxiv.org/abs/1502.03167
//
// Normalization is done as:
// 	γ(x - μ) / σ + β
// γ is the scaling factor and β is the offset factor. These are created by BatchNorm()
type BatchNormOp struct {
	momentum float64 // momentum for the moving average
	epsilon  float64 // small variance to be added to avoid dividing by 0
	dims     int     // 2 or 4. defaults to 4

	// learnables
	runningMean, runningVariance *tensor.Dense

	saveMean, saveVariance *tensor.Dense

	// training? if training then update movingMean and movingVar
	training bool
}

// Arity returns 3
func (op *BatchNormOp) Arity() int { return 3 }

// Type ...
func (op *BatchNormOp) Type() hm.Type {
	dims := op.dims
	if dims == 0 {
		dims = 4 // default to 4 if not set
	}

	t := TensorType{Dims: dims, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, t, t, t)
}

// InferShape from the input values
func (op *BatchNormOp) InferShape(ns ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(ns)); err != nil {
		return nil, errors.Wrapf(err, "batchNorm")
	}

	return ns[0].(tensor.Shape).Clone(), nil
}

// Do performs the batchnorm computation on the values
func (op *BatchNormOp) Do(values ...Value) (retVal Value, err error) {
	if err := checkArity(op, len(values)); err != nil {
		return nil, errors.Wrapf(err, "batchNorm Do")
	}

	var v, out, scale, bias Value
	v = values[0]
	if out, err = CloneValue(v); err != nil {
		return nil, err
	}

	scale = values[1]
	bias = values[2]

	return op.UsePreallocDo(out, v, scale, bias)
}

// ReturnsPtr is true
func (op *BatchNormOp) ReturnsPtr() bool { return true }

// CallsExtern is false
func (op *BatchNormOp) CallsExtern() bool { return false }

// OverwritesInput is -1 (operator doesn't overwrite any input value)
func (op *BatchNormOp) OverwritesInput() int { return -1 }

// WriteHash ...
func (op *BatchNormOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "batchnorm-%1.1f-%1.1f", op.momentum, op.epsilon)
}

// Hashcode ...
func (op *BatchNormOp) Hashcode() uint32 { return simpleHash(op) }

func (op *BatchNormOp) String() string {
	return fmt.Sprintf("batchnorm-%1.1f-%1.1f", op.momentum, op.epsilon)
}

// DoDiff does the gradient computation
func (op *BatchNormOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	diff := &batchnormDiffOp{op}
	xdv, ydv := getDV(inputs[0], output)
	sdv, bdv := getDV(inputs[1], inputs[2])
	_, err := diff.UsePreallocDo(xdv.d, xdv.Value, ydv.d, sdv.Value, bdv.Value, sdv.d, bdv.d)
	return err
}

// DiffWRT ...
func (op *BatchNormOp) DiffWRT(inputs int) []bool { return []bool{true, true, true} }

// SymDiff ...
func (op *BatchNormOp) SymDiff(inputs Nodes, output *Node, grad *Node) (retVal Nodes, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}
	input := inputs[0]
	scale := inputs[1]
	bias := inputs[2]

	g := input.Graph()

	diff := &batchnormDiffOp{op}
	scaleDiff := NewUniqueNode(WithType(scale.Type()), WithShape(scale.Shape().Clone()...), WithChildren(Nodes{scale}), In(g), WithOp(Iop{}))

	biasDiff := NewUniqueNode(WithType(bias.Type()), WithShape(bias.Shape().Clone()...), WithChildren(Nodes{bias}), In(g), WithOp(Iop{}))

	var dy *Node
	if dy, err = ApplyOp(diff, input, grad, scale, bias, scaleDiff, biasDiff); err != nil {
		return nil, err
	}

	return Nodes{dy, scaleDiff, biasDiff}, nil
}

// UsePreallocDo ...
func (op *BatchNormOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	v := inputs[0]
	scale := inputs[1]
	bias := inputs[2]
	switch v.Dtype() {
	case Float64:
		err = op.f64s(v.(*tensor.Dense), prealloc.(*tensor.Dense), scale.(*tensor.Dense), bias.(*tensor.Dense))
	case Float32:
		err = op.f32s(v.(*tensor.Dense), prealloc.(*tensor.Dense), scale.(*tensor.Dense), bias.(*tensor.Dense))
	default:
		return nil, nyi("BatchNorm Do", v.Dtype())
	}

	return prealloc, err
}

// Stats returns the running mean and running variance
func (op *BatchNormOp) Stats() (runningMean tensor.Tensor, runningVariance tensor.Tensor) {
	return op.runningMean, op.runningVariance
}

// SetStats sets the running mean and running variance. The given values are copied
func (op *BatchNormOp) SetStats(runningMean tensor.Tensor, runningVariance tensor.Tensor) error {
	if !runningMean.Shape().Eq(op.runningMean.Shape()) {
		return fmt.Errorf("invalid runningMean shape %v. Expected: %v", runningMean.Shape(), op.runningMean.Shape())
	}

	if runningMean.Dtype() != op.runningMean.Dtype() {
		return fmt.Errorf("invalid runningMean type %v. Expected: %v", runningMean.Dtype(), op.runningMean.Dtype())
	}

	if !runningVariance.Shape().Eq(op.runningVariance.Shape()) {
		return fmt.Errorf("invalid runningVariance shape %v. Expected: %v", runningMean.Shape(), op.runningMean.Shape())
	}

	if runningVariance.Dtype() != op.runningVariance.Dtype() {
		return fmt.Errorf("invalid runningVariance type %v. Expected: %v", runningMean.Dtype(), op.runningMean.Dtype())
	}

	switch op.runningMean.Dtype() {
	case Float32:
		copy(op.runningMean.Data().([]float32), runningMean.Data().([]float32))
		copy(op.runningVariance.Data().([]float32), runningVariance.Data().([]float32))
	case Float64:
		copy(op.runningMean.Data().([]float64), runningMean.Data().([]float64))
		copy(op.runningVariance.Data().([]float64), runningVariance.Data().([]float64))
	}

	return nil
}

// SetTraining configure the op for training mode.
// A call to this function with `true` implicitly calls the Reset() method
func (op *BatchNormOp) SetTraining(isTraining bool) error {
	if isTraining {
		op.Reset()
	}

	op.training = isTraining

	return nil
}

// Reset the operator by zeroing the internals scratch spaces
func (op *BatchNormOp) Reset() error {
	dt := op.runningMean.Dtype()
	var uno interface{}
	switch dt {
	case Float64:
		uno = float64(1)
	case Float32:
		uno = float32(1)
	}

	if err := op.runningVariance.Memset(uno); err != nil {
		return err
	}

	op.runningMean.Zero()

	return nil
}

func (op *BatchNormOp) updateStatsF64(batchSize, channels, spatialDim int, inputT *tensor.Dense) (saveMean []float64, saveVar []float64) {
	momentum := float64(op.momentum)

	inputA := inputT.Float64s()

	op.saveMean.Zero()
	op.saveVariance.Zero()

	saveMean = op.saveMean.Float64s()
	saveVar = op.saveVariance.Float64s()

	runningMean := op.runningMean.Float64s()
	runningVar := op.runningVariance.Float64s()
	n := spatialDim * batchSize

	// NOTE: this can be parallelized by channel, should we?
	if spatialDim == 1 { // image size = 1
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				i := s*channels + c

				saveMean[c] += inputA[i]
			}

			saveMean[c] /= float64(n)

			for s := 0; s < batchSize; s++ {
				i := s*channels + c

				saveVar[c] += (inputA[i] - saveMean[c]) * (inputA[i] - saveMean[c])
			}

			runningMean[c] = (momentum*saveMean[c] + (1-momentum)*runningMean[c])

			unbiasedVar := saveVar[c] / float64(n-1)
			runningVar[c] = (momentum*unbiasedVar + (1-momentum)*runningVar[c])
		}
	} else { // image size > 1
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					saveMean[c] += inputA[i]
				}
			}

			saveMean[c] /= float64(n)

			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					x := inputA[i]

					saveVar[c] += (x - saveMean[c]) * (x - saveMean[c])
				}
			}

			runningMean[c] = (momentum*saveMean[c] + (1-momentum)*runningMean[c])

			unbiasedVar := saveVar[c] / float64(n-1)
			runningVar[c] = (momentum*unbiasedVar + (1-momentum)*runningVar[c])
		}
	}

	return saveMean, saveVar
}

// alpha = scale / sqrt(variance+eps)
// beta = bias - mean * alpha
func (op *BatchNormOp) calculateAlphaAndBetaF64(batchSize, channels, spatialDim int, scaleT, biasT *tensor.Dense, saveMean, saveVar []float64) (alpha []float64, beta []float64) {
	runningMean := op.runningMean.Float64s()
	runningVar := op.runningVariance.Float64s()
	n := spatialDim * batchSize

	scale := scaleT.Float64s()
	bias := biasT.Float64s()

	alpha = make([]float64, channels)
	beta = make([]float64, channels)

	for c := 0; c < channels; c++ {
		var invStd, mean float64

		if op.training {
			mean = saveMean[c]
			invStd = 1 / math.Sqrt(saveVar[c]/float64(n)+float64(op.epsilon))
		} else {
			mean = runningMean[c]
			invStd = 1 / math.Sqrt(runningVar[c]+float64(op.epsilon))
		}

		alpha[c] = invStd * scale[c]
		beta[c] = bias[c] - mean*alpha[c]
	}

	return alpha, beta
}

func (op *BatchNormOp) f64s(input, output, scale, bias *tensor.Dense) (err error) {
	batchSize := input.Shape()[0]
	channels := input.Shape()[1]
	nc := channels * batchSize
	spatialDim := input.Shape().TotalSize() / nc

	var alpha, beta []float64

	if op.training {
		saveMean, saveVar := op.updateStatsF64(batchSize, channels, spatialDim, input)
		alpha, beta = op.calculateAlphaAndBetaF64(batchSize, channels, spatialDim, scale, bias, saveMean, saveVar)
	} else {
		saveMean := make([]float64, channels)
		saveVar := make([]float64, channels)

		alpha, beta = op.calculateAlphaAndBetaF64(batchSize, channels, spatialDim, scale, bias, saveMean, saveVar)
	}

	// output = input * alpha + beta
	outputF64s := output.Float64s()

	if spatialDim == 1 {
		for s := 0; s < batchSize; s++ {
			for c := 0; c < channels; c++ {
				i := s*channels + c

				outputF64s[i] = outputF64s[i]*alpha[c] + beta[c]
			}
		}
	} else {
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					outputF64s[i] = outputF64s[i]*alpha[c] + beta[c]
				}
			}
		}
	}

	return nil
}

func (op *BatchNormOp) updateStatsF32(batchSize, channels, spatialDim int, inputT *tensor.Dense) (saveMean []float32, saveVar []float32) {
	momentum := float32(op.momentum)

	inputA := inputT.Float32s()

	op.saveMean.Zero()
	op.saveVariance.Zero()

	saveMean = op.saveMean.Float32s()
	saveVar = op.saveVariance.Float32s()

	runningMean := op.runningMean.Float32s()
	runningVar := op.runningVariance.Float32s()
	n := spatialDim * batchSize

	// NOTE: this can be parallelized by channel, should we?
	if spatialDim == 1 { // image size = 1
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				i := s*channels + c

				saveMean[c] += inputA[i]
			}

			saveMean[c] /= float32(n)

			for s := 0; s < batchSize; s++ {
				i := s*channels + c

				saveVar[c] += (inputA[i] - saveMean[c]) * (inputA[i] - saveMean[c])
			}

			runningMean[c] = (momentum * saveMean[c]) + (1-momentum)*runningMean[c]

			unbiasedVar := (saveVar[c]) / float32(n-1)
			runningVar[c] = (momentum*unbiasedVar + (1-momentum)*(runningVar[c]))
		}
	} else { // image size > 1
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					saveMean[c] += inputA[i]
				}
			}

			saveMean[c] /= float32(n)

			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					x := inputA[i]

					saveVar[c] += (x - saveMean[c]) * (x - saveMean[c])
				}
			}

			runningMean[c] = (momentum*saveMean[c] + (1-momentum)*runningMean[c])

			unbiasedVar := saveVar[c] / float32(n-1)
			runningVar[c] = momentum*unbiasedVar + (1-momentum)*float32(runningVar[c])
		}
	}

	return saveMean, saveVar
}

// alpha = scale / sqrt(variance+eps)
// beta = bias - mean * alpha
func (op *BatchNormOp) calculateAlphaAndBetaF32(batchSize, channels, spatialDim int, scaleT, biasT *tensor.Dense, saveMean, saveVar []float32) (alpha []float32, beta []float32) {
	runningMean := op.runningMean.Float32s()
	runningVar := op.runningVariance.Float32s()

	n := spatialDim * batchSize

	scale := scaleT.Float32s()
	bias := biasT.Float32s()

	alpha = make([]float32, channels)
	beta = make([]float32, channels)

	for c := 0; c < channels; c++ {
		var invStd, mean float32

		if op.training {
			mean = saveMean[c]
			invStd = 1 / math32.Sqrt(saveVar[c]/float32(n)+float32(op.epsilon))
		} else {
			mean = runningMean[c]
			invStd = 1 / math32.Sqrt(runningVar[c]+float32(op.epsilon))
		}

		alpha[c] = (invStd * scale[c])
		beta[c] = (bias[c] - mean*alpha[c])
	}

	return alpha, beta
}

func (op *BatchNormOp) f32s(input, output, scale, bias *tensor.Dense) (err error) {
	batchSize := input.Shape()[0]
	channels := input.Shape()[1]
	nc := channels * batchSize
	spatialDim := input.Shape().TotalSize() / nc

	var alpha, beta []float32

	if op.training {
		saveMean, saveVar := op.updateStatsF32(batchSize, channels, spatialDim, input)
		alpha, beta = op.calculateAlphaAndBetaF32(batchSize, channels, spatialDim, scale, bias, saveMean, saveVar)
	} else {
		saveMean := make([]float32, channels)
		saveVar := make([]float32, channels)

		alpha, beta = op.calculateAlphaAndBetaF32(batchSize, channels, spatialDim, scale, bias, saveMean, saveVar)
	}

	// output = input * alpha + beta
	outputF32s := output.Float32s()

	if spatialDim == 1 {
		for s := 0; s < batchSize; s++ {
			for c := 0; c < channels; c++ {
				i := s*channels + c

				outputF32s[i] = outputF32s[i]*alpha[c] + beta[c]
			}
		}
	} else {
		for c := 0; c < channels; c++ {
			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					outputF32s[i] = outputF32s[i]*alpha[c] + beta[c]
				}
			}
		}
	}

	return nil
}

type batchnormDiffOp struct{ *BatchNormOp }

func (op *batchnormDiffOp) Arity() int { return 6 }

func (op *batchnormDiffOp) Type() hm.Type {
	dims := op.dims
	if dims == 0 {
		dims = 4
	}

	t := TensorType{Dims: dims, Of: hm.TypeVariable('a')}
	return hm.NewFnType(t, t, t, t, t, t, t)
}

func (op *batchnormDiffOp) InferShape(ns ...DimSizer) (tensor.Shape, error) {
	if err := checkArity(op, len(ns)); err != nil {
		return nil, errors.Wrapf(err, "batchNorm")
	}

	originalShape := ns[0].(tensor.Shape).Clone()

	return originalShape, nil
}

func (op *batchnormDiffOp) Do(values ...Value) (Value, error) {
	input := values[0].(*tensor.Dense)
	grad := values[1].(*tensor.Dense)
	scale := values[2].(*tensor.Dense)
	bias := values[3].(*tensor.Dense)
	scaleDiff := values[4].(*tensor.Dense)
	biasDiff := values[5].(*tensor.Dense)

	dy, err := CloneValue(input)
	if err != nil {
		return nil, err
	}

	v, err := op.UsePreallocDo(dy, input, grad, scale, bias, scaleDiff, biasDiff)

	return v, err
}

// ReturnsPtr is the same exact characteristics of batchnorm
// CallsExtern is the same exact characteristics of batchnorm
// OverwritesInput is the same exact characteristics of batchnorm

func (op *batchnormDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "batchnormdiff-%1.1f-%1.1f", op.momentum, op.epsilon)
}

func (op *batchnormDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *batchnormDiffOp) String() string {
	return fmt.Sprintf("batchnormdiff-%1.1f-%1.1f", op.momentum, op.epsilon)
}

func (op *batchnormDiffOp) DiffWRT(inputs int) []bool {
	// god help those who want to  do 2nd order differentiation on batchnorm
	return []bool{false, false, false, false, false, false}
}

func (op *batchnormDiffOp) SymDiff(inputs Nodes, output *Node, grad *Node) (retVal Nodes, err error) {
	// god help those who want to  do 2nd order differentiation on batchnorm
	return nil, nyi("SymDiff", "batchNormDiffOp")
}

func (op *batchnormDiffOp) DoDiff(ctx ExecutionContext, inputs Nodes, output *Node) error {
	// god help those who want to  do 2nd order differentiation on batchnorm
	return nyi("DoDiff", "batchnormDiffOp")
}

func (op *batchnormDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (retVal Value, err error) {
	input := inputs[0].(*tensor.Dense)
	buffer := prealloc.(*tensor.Dense)
	outGrad := inputs[1].(*tensor.Dense)
	scale := inputs[2].(*tensor.Dense)
	bias := inputs[3].(*tensor.Dense)
	scaleDiff := inputs[4].(*tensor.Dense)
	biasDiff := inputs[5].(*tensor.Dense)

	// log.Printf("out grad: %v", outGrad)

	switch input.Dtype() {
	case Float64:
		err = op.f64s(input, buffer, outGrad, scale, bias, scaleDiff, biasDiff)
	case Float32:
		err = op.f32s(input, buffer, outGrad, scale, bias, scaleDiff, biasDiff)
	default:
		return nil, nyi("batchnormDiffOp", "Do")
	}

	return prealloc, err
}

func (op *batchnormDiffOp) f64s(input, prealloc, outGrad, scaleT, biasT, scaleDiffT, biasDiffT *tensor.Dense) (err error) {
	in := input.Float64s()
	ig := prealloc.Float64s()

	dy := outGrad.Float64s()

	scale := scaleT.Float64s()

	scaleDiff := scaleDiffT.Float64s()
	biasDiff := biasDiffT.Float64s()

	saveMean := op.saveMean.Float64s()
	saveVariance := op.saveVariance.Float64s()
	runningVar := op.runningVariance.Float64s()
	runningMean := op.runningVariance.Float64s()

	batchSize := input.Shape()[0]
	channels := input.Shape()[1]
	nc := batchSize * channels
	spatialDim := len(in) / nc
	n := batchSize * spatialDim

	if spatialDim == 1 {
		for c := 0; c < channels; c++ {
			dotp := float64(0.0)
			sum := 0.0

			var mean, invstd float64
			if op.training {
				mean = saveMean[c]
				invstd = 1 / math.Sqrt(saveVariance[c]/float64(n)+op.epsilon)
			} else {
				mean = runningMean[c]
				invstd = 1 / math.Sqrt(runningVar[c]+op.epsilon)
			}

			for s := 0; s < n; s++ {
				i := s*channels + c

				sum += dy[i]
				partialDotp := (in[i] - mean) * float64(dy[i])
				dotp += partialDotp
			}

			// grad_mean = dySum / N
			gradMean := sum / float64(n)

			if op.training {
				k := float64(dotp*invstd*invstd) / float64(n)

				for s := 0; s < n; s++ {
					i := s*channels + c

					// dx = (x - mean) * k
					ig[i] = ((in[i] - mean) * k)

					// dx = (dy - dx - grad_mean) / variance
					ig[i] = (float64(dy[i]-ig[i]-gradMean) * invstd * scale[c])
				}

				scaleDiff[c] = (dotp * invstd)
			} else {
				for s := 0; s < n; s++ {
					i := s*channels + c

					ig[i] = dy[i] * invstd * scale[c]
				}

				scaleDiff[c] = (dotp * invstd)
			}

			biasDiff[c] = sum
		}
	} else {
		for c := 0; c < channels; c++ {
			dotp := float64(0.0)
			sum := 0.0

			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					sum += dy[i]

					partialDotp := (float64(in[i])*dy[i] - float64(saveMean[c])*float64(dy[i]))
					dotp += partialDotp
				}
			}

			invstd := 1 / math.Sqrt(float64(saveVariance[c])/float64(n)+float64(op.epsilon))
			k := float64(dotp*invstd*invstd) / float64(n)

			// grad_mean = dySum / N
			gradMean := sum / float64(n)

			if op.training {
				for s := 0; s < batchSize; s++ {
					for d := 0; d < spatialDim; d++ {
						i := s*channels*spatialDim + c*spatialDim + d

						// dx = (x - mean) * k
						ig[i] = (in[i] - saveMean[c]) * k

						// dx = (dy - dx - grad_mean) / variance
						ig[i] = (dy[i] - gradMean - ig[i]) * invstd * scale[c]
					}
				}
			} else {
				for s := 0; s < batchSize; s++ {
					for d := 0; d < spatialDim; d++ {
						i := s*channels*spatialDim + c*spatialDim + d

						ig[i] = dy[i] * invstd * scale[c]
					}
				}
			}

			scaleDiff[c] = (dotp * invstd)
			biasDiff[c] = sum
		}
	}

	return nil
}

func (op *batchnormDiffOp) f32s(input, prealloc, outGrad, scaleT, biasT, scaleDiffT, biasDiffT *tensor.Dense) (err error) {
	in := input.Float32s()
	ig := prealloc.Float32s()

	dy := outGrad.Float32s()

	scale := scaleT.Float32s()

	scaleDiff := scaleDiffT.Float32s()
	biasDiff := biasDiffT.Float32s()

	saveMean := op.saveMean.Float32s()
	saveVariance := op.saveVariance.Float32s()
	runningVar := op.runningVariance.Float32s()
	runningMean := op.runningVariance.Float32s()

	batchSize := input.Shape()[0]
	channels := input.Shape()[1]
	nc := batchSize * channels
	spatialDim := len(in) / nc
	n := batchSize * spatialDim

	if spatialDim == 1 {
		for c := 0; c < channels; c++ {
			dotp := float32(0.0)
			sum := float32(0.0)

			var mean, invstd float32
			if op.training {
				mean = saveMean[c]
				invstd = 1 / math32.Sqrt(saveVariance[c]/float32(n)+float32(op.epsilon))
			} else {
				mean = runningMean[c]
				invstd = 1 / math32.Sqrt(runningVar[c]+float32(op.epsilon))
			}

			for s := 0; s < n; s++ {
				i := s*channels + c

				sum += dy[i]
				partialDotp := (in[i] - mean) * dy[i]
				dotp += partialDotp
			}

			// grad_mean = dySum / N
			gradMean := sum / float32(n)

			if op.training {
				k := float32(dotp*invstd*invstd) / float32(n)

				for s := 0; s < n; s++ {
					i := s*channels + c

					// dx = (x - mean) * k
					ig[i] = ((in[i] - mean) * k)

					// dx = (dy - dx - grad_mean) / variance
					ig[i] = (dy[i] - ig[i] - gradMean) * invstd * scale[c]
				}

				scaleDiff[c] = (dotp * invstd)
			} else {
				for s := 0; s < n; s++ {
					i := s*channels + c

					ig[i] = dy[i] * invstd * scale[c]
				}

				scaleDiff[c] = (dotp * invstd)
			}

			biasDiff[c] = sum
		}
	} else {
		for c := 0; c < channels; c++ {
			dotp := float32(0.0)
			sum := float32(0.0)

			for s := 0; s < batchSize; s++ {
				for d := 0; d < spatialDim; d++ {
					i := s*channels*spatialDim + c*spatialDim + d

					sum += dy[i]

					partialDotp := (in[i]*dy[i] - saveMean[c]*dy[i])
					dotp += partialDotp
				}
			}

			invstd := 1 / math32.Sqrt(saveVariance[c]/float32(n)+float32(op.epsilon))
			k := float32(dotp*invstd*invstd) / float32(n)

			// grad_mean = dySum / N
			gradMean := sum / float32(n)

			if op.training {
				for s := 0; s < batchSize; s++ {
					for d := 0; d < spatialDim; d++ {
						i := s*channels*spatialDim + c*spatialDim + d

						// dx = (x - mean) * k
						ig[i] = (in[i] - saveMean[c]) * k

						// dx = (dy - dx - grad_mean) / variance
						ig[i] = (dy[i] - gradMean - ig[i]) * invstd * scale[c]
					}
				}
			} else {
				for s := 0; s < batchSize; s++ {
					for d := 0; d < spatialDim; d++ {
						i := s*channels*spatialDim + c*spatialDim + d

						ig[i] = dy[i] * invstd * scale[c]
					}
				}
			}

			scaleDiff[c] = (dotp * invstd)
			biasDiff[c] = sum
		}
	}

	return nil
}

type globalAveragePoolOp struct{}

func (g *globalAveragePoolOp) Arity() int {
	return 1
}

func (g *globalAveragePoolOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	t := newTensorType(4, a)
	return hm.NewFnType(t, t)
}

func (g *globalAveragePoolOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	b, err := inputs[0].DimSize(0)
	if err != nil {
		return nil, err
	}
	c, err := inputs[0].DimSize(1)
	if err != nil {
		return nil, err
	}
	// check if the shape is correct without doing type inference
	if _, err := inputs[0].DimSize(2); err != nil {
		return nil, err
	}
	if _, err := inputs[0].DimSize(3); err != nil {
		return nil, err
	}
	return tensor.Shape{b, c, 1, 1}, nil
}

func (g *globalAveragePoolOp) Do(inputs ...Value) (Value, error) {
	im := inputs[0]
	switch im.(type) {
	case tensor.Tensor:
		v := im.(tensor.Tensor)
		B, C, H, W := v.Shape()[0], v.Shape()[1], v.Shape()[2], v.Shape()[3]
		s, err := g.InferShape(v.Shape())
		if err != nil {
			return nil, err
		}
		output := tensor.New(tensor.Of(v.Dtype()), tensor.WithShape(s...))
		switch v.Dtype() {
		case tensor.Float64:
			for b := 0; b < B; b++ {
				for c := 0; c < C; c++ {
					var sum float64
					for h := 0; h < H; h++ {
						for w := 0; w < W; w++ {
							val, err := v.At(b, c, h, w)
							if err != nil {
								return nil, err
							}
							sum += val.(float64)
						}
					}
					err := output.SetAt(sum/float64(H*W), b, c, 0, 0)
					if err != nil {
						return nil, err
					}
				}
			}
		case tensor.Float32:
			for b := 0; b < B; b++ {
				for c := 0; c < C; c++ {
					var sum float32
					for h := 0; h < H; h++ {
						for w := 0; w < W; w++ {
							val, err := v.At(b, c, h, w)
							if err != nil {
								return nil, err
							}
							sum += val.(float32)
						}
					}
					err := output.SetAt(sum/float32(H*W), b, c, 0, 0)
					if err != nil {
						return nil, err
					}
				}
			}
		default:
			return nil, nyi("Global Average Pool", v.Dtype())
		}

		return output, nil

	default:
		return nil, nyi("globalAveragePoolOp", inputs)
	}
}

func (g *globalAveragePoolOp) ReturnsPtr() bool {
	return false
}

func (g *globalAveragePoolOp) CallsExtern() bool {
	return false
}

func (g *globalAveragePoolOp) OverwritesInput() int {
	return -1
}

func (g *globalAveragePoolOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "GlobalAveragePool")
}

func (g *globalAveragePoolOp) Hashcode() uint32 {
	return simpleHash(g)
}

func (g *globalAveragePoolOp) String() string {
	return "GlobalAveragePool"
}
