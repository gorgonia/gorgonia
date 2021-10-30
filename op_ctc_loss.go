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
func CTCLoss(logProbs, targets, inputLenghts, targetLengths *Node, reduction Reduction) (*Node, error) {
	op := newCTCLossOp(logProbs.Dtype(), targets.Shape().Dims(), reduction)

	negLogs, err := ApplyOp(op, logProbs, targets, inputLenghts, targetLengths)
	if err != nil {
		return nil, err
	}

	return negLogs, nil // FIXME
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
		return nil, fmt.Errorf("invalid type %v for inputLenghts. it should be Int", inputLengthsT.Dtype())
	}

	targetLengthsT := inputs[3].(*tensor.Dense)
	if targetLengthsT.Dtype() != tensor.Int {
		return nil, fmt.Errorf("invalid type %v for inputLenghts. it should be Int", targetLengthsT.Dtype())
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

	maxInputLenght := logProbsT.Shape()[0]
	for i := 0; i < batchSize; i++ {
		if inputLengths[i] > maxInputLenght {
			return fmt.Errorf("expected inputLenghts to have value at most %v, but got %v", maxInputLenght, inputLengths[i])
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

	maxInputLenght := logProbsT.Shape()[0]
	for i := 0; i < batchSize; i++ {
		if inputLengths[i] > maxInputLenght {
			return fmt.Errorf("expected inputLenghts to have value at most %v, but got %v", maxInputLenght, inputLengths[i])
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
	input := inputs[1]
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

	log.Printf("grad out: %v", gradOutT)

	targets := targetsT.Ints()
	targetLengths := targetLengthsT.Ints()
	inputLengths := inputLengthsT.Ints()

	log.Printf("input lenghts: %v", inputLengths)

	_ = targets
	_ = inputLengths

	inputSize := logProbsT.Shape()[0] // rows
	batchSize := logProbsT.Shape()[1] // blocks
	numLabels := logProbsT.Shape()[2] // columns
	spatialDim := inputSize * numLabels
	logAlphaSpatialDim := tensor.Shape(op.logAlpha.Shape()[1:]).TotalSize()

	log.Printf("spatial dim: %v", spatialDim)

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

	log.Printf("max target l: %v", maxTargetLength)
	log.Printf("offset: %v", targetBatchOffsets)
	log.Printf("stride: %v", targetStride)

	negInf := math.Inf(-1)

	gradT := tensor.New(tensor.Of(op.dtype), tensor.WithShape(logProbsT.Shape()...))
	if err := gradT.Memset(negInf); err != nil {
		return nil, err
	}

	logBetaT := tensor.New(tensor.WithShape(op.logAlpha.Shape()...), tensor.Of(op.logAlpha.Dtype()))
	if err := logBetaT.Memset(negInf); err != nil {
		return nil, err
	}

	lppT, err := tensor.Transpose(logProbsT, 1, 0, 2)
	if err != nil {
		return nil, err
	}

	_ = logBetaT
	_ = lppT

	err = gradT.T(1, 0, 2)
	if err != nil {
		return nil, err
	}

	negLogLikelihood := op.negLogLikelihood.Float64s()
	logBeta := logBetaT.Float64s()
	logAlpha := op.logAlpha.Float64s()
	logAlphaWidth := 2*maxTargetLength + 1
	lpp := lppT.(*tensor.Dense).Float64s()
	// gradOut := gradOutT.Float64s()

	_ = logAlphaWidth

	log.Printf("neglog: %v", op.negLogLikelihood)

	// this can be parallelized
	for b := 0; b < batchSize; b++ {
		log.Printf("START BATCH: %v", b)
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetsOffset := targetBatchOffsets[b]
		targetWidth := 2*targetLength + 1

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lpp[initialIndex:finalIndex]
		gradSlice, err := gradT.Slice(S(b))
		if err != nil {
			return nil, err
		}

		nll := negLogLikelihood[b]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlpha[initialLogAlphaIndex:finalLogAlphaIndex]
		logBetaSection := logBeta[initialLogAlphaIndex:finalLogAlphaIndex]

		_ = logAlphaSection
		_ = targetWidth

		log.Printf("nll: %v", nll)
		log.Printf("grad slice: %v", gradSlice)
		_ = targetsOffset

		log.Printf("target width: %v num labels: %v", targetWidth, numLabels)

		if inputLength > 0 {
			// log.Printf("lpp: %v", lppSection)

			// log.Printf("input lenght: %v", inputLength)
			logBetaSection[(inputLength-1)*targetWidth+2*targetLength] = lppSection[(inputLength-1)*numLabels]
			// log.Printf("logBetaSection = %v", logBetaSection[(inputLength-1)*targetWidth+2*targetLength])

			gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+2*targetLength]+logBetaSection[(inputLength-1)*targetWidth+2*targetLength], inputLength-1, 0)

			log.Printf("[1] set %v,%v to %v", inputLength-1, 0, logAlphaSection[(inputLength-1)*targetWidth+2*targetLength]+logBetaSection[(inputLength-1)*targetWidth+2*targetLength])

			// gradSection[(inputLength-1)*numLabels] = logAlphaSection[(inputLength-1)*targetWidth+2*targetLength] + logBetaSection[(inputLength-1)*targetWidth+2*targetLength]
			// log.Printf("grad = %v", gradSlice)

			// log.Printf("%v", gradT)
			// log.Printf("%v", logBetaT)

			if targetLength > 0 {
				currentPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, 2*targetLength-1)

				logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)] = lppSection[(inputLength-1)*numLabels+currentPrime]

				gradSlice.SetAt(logAlphaSection[(inputLength-1)*targetWidth+(2*targetLength-1)]+logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)], (inputLength - 1), currentPrime)

				log.Printf("[2] set %v,%v to %v", inputLength-1, currentPrime, logAlphaSection[(inputLength-1)*targetWidth+(2*targetLength-1)]+logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)])

				// gradSection[(inputLength-1)*numLabels+currentPrime] = logAlphaSection[(inputLength-1)*targetWidth+(2*targetLength-1)] + logBetaSection[(inputLength-1)*targetWidth+(2*targetLength-1)]

				// log.Printf("%v", gradSection[(inputLength-1)*numLabels+currentPrime])
			}

			for t := inputLength - 2; t >= 0; t-- {
				for s := 2 * targetLength; s >= 0; s-- {
					baseIndex := (t+1)*targetWidth + s
					lb1 := logBetaSection[baseIndex]
					lbmax := lb1

					var lb2, lb3 float64

					currentTargetPrime := op.getPrimeTarget(targets, targetsOffset, targetStride, s)

					log.Printf("target prime for %v = %v", s, currentTargetPrime)

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

					log.Printf("log alpha: %v + log beta: %v = %v", logAlphaSection[t*targetWidth+s], logBetaSection[t*targetWidth+s], logAlphaBeta)

					lcab := op.getOrPanicF64(gradSlice, t, currentTargetPrime)

					log.Printf("get value from [%v,%v] = %v", t, currentTargetPrime, lcab)

					if lcab == negInf {
						log.Printf("[3.1] set %v, %v to %v", t, s, logAlphaBeta)

						gradSlice.SetAt(logAlphaBeta, t, currentTargetPrime)
					} else {
						max := math.Max(lcab, logAlphaBeta)
						v := math.Log(math.Exp(lcab-max)+math.Exp(logAlphaBeta-max)) + max
						gradSlice.SetAt(
							v,
							t, currentTargetPrime,
						)

						log.Printf("lcab: %v max: %v log alpha beta: %v", lcab, max, logAlphaBeta)

						log.Printf("[3.2] set %v, %v to %v", t, s, v)
					}

					// lcab := &gradSection[t*numLabels+currentTargetPrime]
					// if *lcab == negInf {
					// 	*lcab = logAlphaBeta
					// } else {
					// 	max := math.Max(*lcab, logAlphaBeta)
					// 	*lcab = math.Log(math.Exp(*lcab-max)+math.Exp(logAlphaBeta)) + max
					// }
				}
			}

			gr := 1.0 //gradOut[b]
			for t := 0; t < inputLength; t++ {
				for c := 0; c < numLabels; c++ {
					res := op.getOrPanicF64(gradSlice, t, c)
					lp := lppSection[t*numLabels+c]

					v := (math.Exp(lp) - math.Exp(res+nll-lp)) * gr

					gradSlice.SetAt(v, t, c)
				}
			}
		}
	}

	gradT.UT()
	log.Printf("%v", gradT)

	return prealloc, nil
}

func (op ctcLossDiffOp) getOrPanicF64(view tensor.View, coords ...int) float64 {
	v, err := view.At(coords...)
	if err != nil {
		panic(err)
	}

	return v.(float64)
}

// ensure it complies with the Op interface
var (
	_ Op = &ctcLossDiffOp{}
)
