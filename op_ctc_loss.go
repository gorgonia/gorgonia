package gorgonia

import (
	"fmt"
	"hash"
	"log"
	"math"

	"github.com/chewxy/hm"
	"github.com/chewxy/math32"
	"gorgonia.org/tensor"
)

type Reduction uint

const (
	ReductionMean Reduction = iota
	ReductionSum
)

// CTCLoss -  implements the ctc loss operation
// This is the implementation of the following paper: http://www.cs.toronto.edu/~graves/icml_2006.pdf
func CTCLoss(logProbs, targets, inputLengths, targetLengths *Node, reduction Reduction) (*Node, error) {
	op := newCTCLossOp(logProbs.Dtype(), targets.Shape().Dims(), reduction)

	output, err := ApplyOp(op, logProbs, targets, inputLengths, targetLengths)
	if err != nil {
		return nil, err
	}

	return output, nil
}

type ctcLossOp struct {
	dtype      tensor.Dtype
	targetDims int
	reduction  Reduction

	logAlpha         *tensor.Dense
	negLogLikelihood *tensor.Dense
}

func newCTCLossOp(dtype tensor.Dtype, targetDims int, reduction Reduction) *ctcLossOp {
	op := &ctcLossOp{
		dtype:      dtype,
		targetDims: targetDims,
		reduction:  reduction,
	}

	return op
}

func (op *ctcLossOp) Arity() int { return 4 }

func (op *ctcLossOp) ReturnsPtr() bool { return false }

func (op *ctcLossOp) CallsExtern() bool { return false }

func (op *ctcLossOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, "CTCLoss{}()")
}

func (op *ctcLossOp) Hashcode() uint32 { return simpleHash(op) }

func (op *ctcLossOp) String() string {
	return fmt.Sprintf("CTCLoss{}()")
}

func (op *ctcLossOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	return tensor.Shape{}, nil
}

func (op *ctcLossOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := makeTensorType(op.targetDims, tensor.Int)
	c := makeTensorType(1, tensor.Int)

	d := op.dtype

	return hm.NewFnType(a, b, c, c, d)
}

func (op *ctcLossOp) OverwritesInput() int { return -1 }

func (op *ctcLossOp) getPrimeTarget(targets []int, offset, stride, idx int) int {
	div, mod := divmod(idx, 2)
	if mod == 0 {
		return 0
	}

	return targets[offset+stride*div]
}

func (op *ctcLossOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	logProbsT := inputs[0].(*tensor.Dense)
	targetsT := inputs[1].(*tensor.Dense)
	if targetsT.Dtype() != tensor.Int {
		return nil, fmt.Errorf("invalid type %v for targets. it should be Int", targetsT.Dtype())
	}

	inputLengthsT := inputs[2].(*tensor.Dense)
	if inputLengthsT.Dtype() != tensor.Int {
		return nil, fmt.Errorf("invalid type %v for inputLengths. it should be Int", inputLengthsT.Dtype())
	}

	targetLengthsT := inputs[3].(*tensor.Dense)
	if targetLengthsT.Dtype() != tensor.Int {
		return nil, fmt.Errorf("invalid type %v for inputLengths. it should be Int", targetLengthsT.Dtype())
	}

	var err error

	switch logProbsT.Dtype() {
	case Float64:
		err = op.f64s(logProbsT, prealloc.(*tensor.Dense), targetsT, inputLengthsT, targetLengthsT)
	case Float32:
		err = op.f32s(logProbsT, prealloc.(*tensor.Dense), targetsT, inputLengthsT, targetLengthsT)
	default:
		return nil, nyi("CTCLoss Do", logProbsT.Dtype())
	}

	return prealloc, err
}

func (op *ctcLossOp) f64s(logProbsT, prealloc, targetsT, inputLengthsT, targetLengthsT *tensor.Dense) error {
	targets := targetsT.Ints()
	targetLengths := targetLengthsT.Ints()
	inputLengths := inputLengthsT.Ints()

	inputSize := logProbsT.Shape()[0] // rows
	batchSize := logProbsT.Shape()[1] // blocks
	numLabels := logProbsT.Shape()[2] // columns
	spatialDim := inputSize * numLabels

	maxTargetLength := 0
	targetStride := 0

	targetBatchOffsets := make([]int, batchSize)
	if targetsT.Dims() == 1 {
		pos := 0
		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = pos
			pos += targetLengths[i]
			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[0]
	} else {
		batchStride := targetsT.Strides()[0]
		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = i * batchStride
			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[1]
	}

	maxInputLength := logProbsT.Shape()[0]
	for i := 0; i < batchSize; i++ {
		if inputLengths[i] > maxInputLength {
			return fmt.Errorf("expected inputLengths to have value at most %v, but got %v", maxInputLength, inputLengths[i])
		}
	}
	negInf := math.Inf(-1)

	logAlphaWidth := 2*maxTargetLength + 1
	logAlpha := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize, logProbsT.Shape()[0], logAlphaWidth),
	)

	logAlphaSpatialDim := tensor.Shape(logAlpha.Shape()[1:]).TotalSize()

	logAlphaView, err := logAlpha.Narrow(1, 0, 1)
	if err != nil {
		return err
	}

	if err := logAlphaView.Memset(negInf); err != nil {
		return err
	}

	negLogLikelihood := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize),
	)

	lpp, err := tensor.Transpose(logProbsT, 1, 0, 2)
	if err != nil {
		return err
	}

	logAlphaA := logAlpha.Float64s()
	lppA := lpp.(*tensor.Dense).Float64s()
	negLogLikelihoodA := negLogLikelihood.Float64s()

	// this can be paralellized
	for b := 0; b < batchSize; b++ {
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetWidth := 2*targetLength + 1
		targetsOffset := targetBatchOffsets[b]

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lppA[initialIndex:finalIndex]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlphaA[initialLogAlphaIndex:finalLogAlphaIndex]

		logAlphaSection[0] = lppSection[0]

		if targetLength > 0 {
			logAlphaSection[1] = lppSection[op.getPrimeTarget(targets, targetsOffset, targetStride, 1)]
		}

		for t := 1; t < inputLength; t++ {
			for s := 0; s < targetWidth; s++ {
				currentTargetPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, s)

				i := (t-1)*(targetWidth) + s
				la1 := logAlphaSection[i]

				lamax := la1
				var la2, la3 float64

				if s > 0 {
					la2 = logAlphaSection[i-1]
					if la2 > lamax {
						lamax = la2
					}
				} else {
					la2 = negInf
				}

				if s > 1 && op.getPrimeTarget(targets, targetsOffset, targetStride, s-2) != currentTargetPrime {
					la3 = logAlphaSection[i-2]
					if la3 > lamax {
						lamax = la3
					}
				} else {
					la3 = negInf
				}

				if lamax == negInf {
					lamax = 0
				}

				logAlphaSection[t*targetWidth+s] = math.Log(math.Exp(la1-lamax)+math.Exp(la2-lamax)+math.Exp(la3-lamax)) + lamax + lppSection[t*numLabels+currentTargetPrime]
			}
		}

		if targetLength == 0 {
			negLogLikelihoodA[b] = logAlphaSection[(inputLength-1)*targetWidth]
		} else {
			l1 := logAlphaSection[(inputLength-1)*targetWidth+targetLength*2]
			l2 := logAlphaSection[(inputLength-1)*targetWidth+targetLength*2-1]
			max := l1
			if l2 > max {
				max = l2
			}

			if max == negInf {
				max = 0
			}

			logLikelihood := math.Log(math.Exp(l1-max)+math.Exp(l2-max)) + max
			negLogLikelihoodA[b] = -logLikelihood
		}
	}

	loss := 0.0

	for i, v := range targetLengths {
		if op.reduction == ReductionSum {
			loss += negLogLikelihoodA[i]
		} else {
			if v < 1 {
				v = 1
			}

			loss += negLogLikelihoodA[i] / float64(v)
		}
	}

	if op.reduction == ReductionMean {
		loss /= float64(len(targetLengths))
	}

	prealloc.Set(0, loss)
	op.logAlpha = logAlpha
	op.negLogLikelihood = negLogLikelihood

	return nil
}

func (op *ctcLossOp) f32s(logProbsT, prealloc, targetsT, inputLengthsT, targetLengthsT *tensor.Dense) error {
	targets := targetsT.Ints()
	targetLengths := targetLengthsT.Ints()
	inputLengths := inputLengthsT.Ints()

	inputSize := logProbsT.Shape()[0] // rows
	batchSize := logProbsT.Shape()[1] // blocks
	numLabels := logProbsT.Shape()[2] // columns
	spatialDim := inputSize * numLabels

	maxTargetLength := 0
	targetStride := 0

	targetBatchOffsets := make([]int, batchSize)
	if targetsT.Dims() == 1 {
		pos := 0
		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = pos
			pos += targetLengths[i]
			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[0]
	} else {
		batchStride := targetsT.Strides()[0]
		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = i * batchStride
			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[1]
	}

	maxInputLength := logProbsT.Shape()[0]
	for i := 0; i < batchSize; i++ {
		if inputLengths[i] > maxInputLength {
			return fmt.Errorf("expected inputLengths to have value at most %v, but got %v", maxInputLength, inputLengths[i])
		}
	}
	negInf := math32.Inf(-1)

	logAlphaWidth := 2*maxTargetLength + 1
	logAlpha := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize, logProbsT.Shape()[0], logAlphaWidth),
	)

	logAlphaSpatialDim := tensor.Shape(logAlpha.Shape()[1:]).TotalSize()

	logAlphaView, err := logAlpha.Narrow(1, 0, 1)
	if err != nil {
		return err
	}

	if err := logAlphaView.Memset(negInf); err != nil {
		return err
	}

	negLogLikelihood := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize),
	)

	lpp, err := tensor.Transpose(logProbsT, 1, 0, 2)
	if err != nil {
		return err
	}

	logAlphaA := logAlpha.Float32s()
	lppA := lpp.(*tensor.Dense).Float32s()
	negLogLikelihoodA := negLogLikelihood.Float32s()

	// this can be paralellized
	for b := 0; b < batchSize; b++ {
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetWidth := 2*targetLength + 1
		targetsOffset := targetBatchOffsets[b]

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lppA[initialIndex:finalIndex]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlphaA[initialLogAlphaIndex:finalLogAlphaIndex]

		logAlphaSection[0] = lppSection[0]

		if targetLength > 0 {
			logAlphaSection[1] = lppSection[op.getPrimeTarget(targets, targetsOffset, targetStride, 1)]
		}

		for t := 1; t < inputLength; t++ {
			for s := 0; s < targetWidth; s++ {
				currentTargetPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, s)

				i := (t-1)*(targetWidth) + s
				la1 := logAlphaSection[i]

				lamax := la1
				var la2, la3 float32

				if s > 0 {
					la2 = logAlphaSection[i-1]
					if la2 > lamax {
						lamax = la2
					}
				} else {
					la2 = negInf
				}

				if s > 1 && op.getPrimeTarget(targets, targetsOffset, targetStride, s-2) != currentTargetPrime {
					la3 = logAlphaSection[i-2]
					if la3 > lamax {
						lamax = la3
					}
				} else {
					la3 = negInf
				}

				if lamax == negInf {
					lamax = 0
				}

				logAlphaSection[t*targetWidth+s] = math32.Log(math32.Exp(la1-lamax)+math32.Exp(la2-lamax)+math32.Exp(la3-lamax)) + lamax + lppSection[t*numLabels+currentTargetPrime]
			}
		}

		if targetLength == 0 {
			negLogLikelihoodA[b] = logAlphaSection[(inputLength-1)*targetWidth]
		} else {
			l1 := logAlphaSection[(inputLength-1)*targetWidth+targetLength*2]
			l2 := logAlphaSection[(inputLength-1)*targetWidth+targetLength*2-1]
			max := l1
			if l2 > max {
				max = l2
			}

			if max == negInf {
				max = 0
			}

			logLikelihood := math32.Log(math32.Exp(l1-max)+math32.Exp(l2-max)) + max
			negLogLikelihoodA[b] = -logLikelihood
		}
	}

	loss := float32(0.0)

	for i, v := range targetLengths {
		if op.reduction == ReductionSum {
			loss += negLogLikelihoodA[i]
		} else {
			if v < 1 {
				v = 1
			}

			loss += negLogLikelihoodA[i] / float32(v)
		}
	}

	if op.reduction == ReductionMean {
		loss /= float32(len(targetLengths))
	}

	prealloc.Set(0, loss)
	op.logAlpha = logAlpha
	op.negLogLikelihood = negLogLikelihood

	return nil
}

func (op *ctcLossOp) Do(inputs ...Value) (retVal Value, err error) {
	logProbsT := inputs[0].(*tensor.Dense)

	prealloc := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(),
	)

	return op.UsePreallocDo(prealloc, inputs...)
}

// SymDiff applies the diff op. Implementation for SDOp interface.
func (op *ctcLossOp) SymDiff(inputs Nodes, output, grad *Node) (Nodes, error) {
	err := checkArity(op, len(inputs))
	if err != nil {
		return nil, err
	}

	logProbs := inputs[0]
	targets := inputs[1]
	inputLengths := inputs[2]
	targetLengths := inputs[3]

	diffOp := &ctcLossDiffOp{op}

	ret, err := ApplyOp(diffOp, logProbs, targets, inputLengths, targetLengths, grad)

	return Nodes{ret, nil, nil, nil, nil}, err
}

// DiffWRT is an implementation for the SDOp interface
func (op *ctcLossOp) DiffWRT(inputs int) []bool {
	return []bool{true, false, false, false, false}
}

type ctcLossDiffOp struct {
	*ctcLossOp
}

func (op *ctcLossDiffOp) Arity() int { return 5 }

func (op *ctcLossDiffOp) WriteHash(h hash.Hash) {
	fmt.Fprintf(h, op.String())
}

func (op *ctcLossDiffOp) Hashcode() uint32 { return simpleHash(op) }

func (op *ctcLossDiffOp) String() string {
	return fmt.Sprintf("ctcLossDiff{}()")
}

func (op *ctcLossDiffOp) InferShape(inputs ...DimSizer) (tensor.Shape, error) {
	s := inputs[0].(tensor.Shape).Clone()

	return s, nil
}

func (op *ctcLossDiffOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := makeTensorType(op.targetDims, tensor.Int)
	c := makeTensorType(1, tensor.Int)
	d := hm.TypeVariable('d')

	return hm.NewFnType(a, b, c, c, d, a)
}

func (op *ctcLossDiffOp) OverwritesInput() int { return -1 }

func (op *ctcLossDiffOp) Do(inputs ...Value) (Value, error) {
	input := inputs[0]
	prealloc := tensor.New(tensor.WithShape(input.Shape().Clone()...), tensor.Of(input.Dtype()))

	return op.UsePreallocDo(prealloc, inputs...)
}

func (op *ctcLossDiffOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
	if err := checkArity(op, len(inputs)); err != nil {
		return nil, err
	}

	logProbsT := inputs[0].(*tensor.Dense)
	targetsT := inputs[1].(*tensor.Dense)
	inputLengthsT := inputs[2].(*tensor.Dense)
	targetLengthsT := inputs[3].(*tensor.Dense)
	gradOutT := inputs[4]

	switch logProbsT.Dtype() {
	case Float64:
		op.f64s(logProbsT, targetsT, inputLengthsT, targetLengthsT, prealloc.(*tensor.Dense), gradOutT.(*F64))
	case Float32:
		op.f32s(logProbsT, targetsT, inputLengthsT, targetLengthsT, prealloc.(*tensor.Dense), gradOutT.(*F32))
	default:
		log.Panicf("%T type is not supported for CTCLoss op", logProbsT.Dtype())
	}

	return prealloc, nil
}

func (op *ctcLossDiffOp) f64s(logProbsT, targetsT, inputLengthsT, targetLengthsT, gradT *tensor.Dense, gradOutT *F64) error {
	targets := targetsT.Ints()
	targetLengths := targetLengthsT.Ints()
	inputLengths := inputLengthsT.Ints()

	inputSize := logProbsT.Shape()[0] // rows
	batchSize := logProbsT.Shape()[1] // blocks
	numLabels := logProbsT.Shape()[2] // columns
	spatialDim := inputSize * numLabels
	logAlphaSpatialDim := tensor.Shape(op.logAlpha.Shape()[1:]).TotalSize()

	maxTargetLength := 0
	targetStride := 0

	targetBatchOffsets := make([]int, batchSize)

	if targetsT.Dims() == 1 {
		pos := 0

		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = pos
			pos += targetLengths[i]

			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[0]
	} else {
		batchStride := targetsT.Strides()[0]

		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = i * batchStride
		}

		targetStride = targetsT.Strides()[1]
		maxTargetLength = targetsT.Shape()[1]
	}

	negInf := math.Inf(-1)
	if err := gradT.Memset(negInf); err != nil {
		return err
	}

	logBetaT := tensor.New(tensor.WithShape(op.logAlpha.Shape()...), tensor.Of(op.logAlpha.Dtype()))
	if err := logBetaT.Memset(negInf); err != nil {
		return err
	}

	lppT, err := tensor.Transpose(logProbsT, 1, 0, 2) // NOTE: I think we can optimize memory usage here
	if err != nil {
		return err
	}

	err = gradT.T(1, 0, 2)
	if err != nil {
		return err
	}

	negLogLikelihood := op.negLogLikelihood.Float64s()
	logBeta := logBetaT.Float64s()
	logAlpha := op.logAlpha.Float64s()
	lpp := lppT.(*tensor.Dense).Float64s()

	// this can be parallelized
	for b := 0; b < batchSize; b++ {
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetsOffset := targetBatchOffsets[b]
		targetWidth := 2*targetLength + 1

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lpp[initialIndex:finalIndex]
		gradSlice, err := gradT.Slice(S(b))
		if err != nil {
			return err
		}

		nll := negLogLikelihood[b]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlpha[initialLogAlphaIndex:finalLogAlphaIndex]
		logBetaSection := logBeta[initialLogAlphaIndex:finalLogAlphaIndex]

		if inputLength > 0 {
			logBetaSection[(inputLength-1)*targetWidth+2*targetLength] = lppSection[(inputLength-1)*numLabels]

			gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+2*targetLength]+logBetaSection[(inputLength-1)*targetWidth+2*targetLength], inputLength-1, 0)

			if targetLength > 0 {
				currentPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, 2*targetLength-1)

				logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)] = lppSection[(inputLength-1)*numLabels+currentPrime]

				gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+(2*targetLength-1)]+logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)], (inputLength - 1), currentPrime)
			}

			for t := inputLength - 2; t >= 0; t-- {
				for s := 2 * targetLength; s >= 0; s-- {
					baseIndex := (t+1)*targetWidth + s
					lb1 := logBetaSection[baseIndex]
					lbmax := lb1

					var lb2, lb3 float64

					currentTargetPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, s)

					if s < 2*targetLength {
						lb2 = logBetaSection[baseIndex+1]
						if lb2 > lbmax {
							lbmax = lb2
						}
					} else {
						lb2 = negInf
					}

					if s < 2*targetLength-1 && op.getPrimeTarget(targets, targetsOffset, targetStride, s+2) != currentTargetPrime {
						lb3 = logBetaSection[baseIndex+2]
						if lb3 > lbmax {
							lbmax = lb3
						}
					} else {
						lb3 = negInf
					}

					if lbmax == negInf {
						lbmax = 0
					}

					logBetaSection[t*targetWidth+s] = math.Log(
						math.Exp(lb1-lbmax)+math.Exp(lb2-lbmax)+math.Exp(lb3-lbmax)) + lbmax + lppSection[t*numLabels+currentTargetPrime]

					logAlphaBeta := logAlphaSection[t*targetWidth+s] + logBetaSection[t*targetWidth+s]

					lcab := op.getOrPanicF64(gradSlice, t, currentTargetPrime)
					if lcab == negInf {
						gradSlice.SetAt(logAlphaBeta, t, currentTargetPrime)
					} else {
						max := math.Max(lcab, logAlphaBeta)
						v := math.Log(math.Exp(lcab-max)+math.Exp(logAlphaBeta-max)) + max
						gradSlice.SetAt(
							v,
							t, currentTargetPrime,
						)
					}
				}
			}

			for t := 0; t < inputLength; t++ {
				for c := 0; c < numLabels; c++ {
					res := op.getOrPanicF64(gradSlice, t, c)
					lp := lppSection[t*numLabels+c]

					v := (math.Exp(lp) - math.Exp(res+nll-lp)) * float64(*gradOutT)

					gradSlice.SetAt(v, t, c)
				}
			}
		}
	}

	gradT.UT()

	return nil
}

func (op *ctcLossDiffOp) f32s(logProbsT, targetsT, inputLengthsT, targetLengthsT, gradT *tensor.Dense, gradOutT *F32) error {
	targets := targetsT.Ints()
	targetLengths := targetLengthsT.Ints()
	inputLengths := inputLengthsT.Ints()

	inputSize := logProbsT.Shape()[0] // rows
	batchSize := logProbsT.Shape()[1] // blocks
	numLabels := logProbsT.Shape()[2] // columns
	spatialDim := inputSize * numLabels
	logAlphaSpatialDim := tensor.Shape(op.logAlpha.Shape()[1:]).TotalSize()

	maxTargetLength := 0
	targetStride := 0

	targetBatchOffsets := make([]int, batchSize)

	if targetsT.Dims() == 1 {
		pos := 0

		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = pos
			pos += targetLengths[i]

			if maxTargetLength < targetLengths[i] {
				maxTargetLength = targetLengths[i]
			}
		}

		targetStride = targetsT.Strides()[0]
	} else {
		batchStride := targetsT.Strides()[0]

		for i := 0; i < batchSize; i++ {
			targetBatchOffsets[i] = i * batchStride
		}

		targetStride = targetsT.Strides()[1]
		maxTargetLength = targetsT.Shape()[1]
	}

	negInf := math32.Inf(-1)
	if err := gradT.Memset(negInf); err != nil {
		return err
	}

	logBetaT := tensor.New(tensor.WithShape(op.logAlpha.Shape()...), tensor.Of(op.logAlpha.Dtype()))
	if err := logBetaT.Memset(negInf); err != nil {
		return err
	}

	lppT, err := tensor.Transpose(logProbsT, 1, 0, 2) // NOTE: I think we can optimize memory usage here
	if err != nil {
		return err
	}

	err = gradT.T(1, 0, 2)
	if err != nil {
		return err
	}

	negLogLikelihood := op.negLogLikelihood.Float32s()
	logBeta := logBetaT.Float32s()
	logAlpha := op.logAlpha.Float32s()
	lpp := lppT.(*tensor.Dense).Float32s()

	// this can be parallelized
	for b := 0; b < batchSize; b++ {
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetsOffset := targetBatchOffsets[b]
		targetWidth := 2*targetLength + 1

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lpp[initialIndex:finalIndex]
		gradSlice, err := gradT.Slice(S(b))
		if err != nil {
			return err
		}

		nll := negLogLikelihood[b]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlpha[initialLogAlphaIndex:finalLogAlphaIndex]
		logBetaSection := logBeta[initialLogAlphaIndex:finalLogAlphaIndex]

		if inputLength > 0 {
			logBetaSection[(inputLength-1)*targetWidth+2*targetLength] = lppSection[(inputLength-1)*numLabels]

			gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+2*targetLength]+logBetaSection[(inputLength-1)*targetWidth+2*targetLength], inputLength-1, 0)

			if targetLength > 0 {
				currentPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, 2*targetLength-1)

				logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)] = lppSection[(inputLength-1)*numLabels+currentPrime]

				gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+(2*targetLength-1)]+logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)], (inputLength - 1), currentPrime)
			}

			for t := inputLength - 2; t >= 0; t-- {
				for s := 2 * targetLength; s >= 0; s-- {
					baseIndex := (t+1)*targetWidth + s
					lb1 := logBetaSection[baseIndex]
					lbmax := lb1

					var lb2, lb3 float32

					currentTargetPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, s)

					if s < 2*targetLength {
						lb2 = logBetaSection[baseIndex+1]
						if lb2 > lbmax {
							lbmax = lb2
						}
					} else {
						lb2 = negInf
					}

					if s < 2*targetLength-1 && op.getPrimeTarget(targets, targetsOffset, targetStride, s+2) != currentTargetPrime {
						lb3 = logBetaSection[baseIndex+2]
						if lb3 > lbmax {
							lbmax = lb3
						}
					} else {
						lb3 = negInf
					}

					if lbmax == negInf {
						lbmax = 0
					}

					logBetaSection[t*targetWidth+s] = math32.Log(
						math32.Exp(lb1-lbmax)+math32.Exp(lb2-lbmax)+math32.Exp(lb3-lbmax)) + lbmax + lppSection[t*numLabels+currentTargetPrime]

					logAlphaBeta := logAlphaSection[t*targetWidth+s] + logBetaSection[t*targetWidth+s]

					lcab := op.getOrPanicF32(gradSlice, t, currentTargetPrime)
					if lcab == negInf {
						gradSlice.SetAt(logAlphaBeta, t, currentTargetPrime)
					} else {
						max := math32.Max(lcab, logAlphaBeta)
						v := math32.Log(math32.Exp(lcab-max)+math32.Exp(logAlphaBeta-max)) + max
						gradSlice.SetAt(
							v,
							t, currentTargetPrime,
						)
					}
				}
			}

			for t := 0; t < inputLength; t++ {
				for c := 0; c < numLabels; c++ {
					res := op.getOrPanicF32(gradSlice, t, c)
					lp := lppSection[t*numLabels+c]

					v := (math32.Exp(lp) - math32.Exp(res+nll-lp)) * float32(*gradOutT)

					gradSlice.SetAt(v, t, c)
				}
			}
		}
	}

	gradT.UT()

	return nil
}

func (op ctcLossDiffOp) getOrPanic(view tensor.View, coords ...int) interface{} {
	v, err := view.At(coords...)
	if err != nil {
		panic(err)
	}

	return v
}

func (op ctcLossDiffOp) getOrPanicF64(view tensor.View, coords ...int) float64 {
	return op.getOrPanic(view, coords...).(float64)
}

func (op ctcLossDiffOp) getOrPanicF32(view tensor.View, coords ...int) float32 {
	return op.getOrPanic(view, coords...).(float32)
}

// ensure it complies with the Op interface
var (
	_ Op = &ctcLossDiffOp{}
)
