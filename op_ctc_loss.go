package gorgonia

import (
	"fmt"
	"hash"
	"log"
	"math"

	"github.com/chewxy/hm"
	"gorgonia.org/tensor"
)

type Reduction uint

const (
	ReductionMean Reduction = iota
	ReductionSum
)

// CTCLoss -  implements the ctc loss operation
// This is the implementation of the following paper: http://www.cs.toronto.edu/~graves/icml_2006.pdf
func CTCLoss(logProbs, targets, inputLenghts, targetLenghts *Node, reduction Reduction) (*Node, error) {
	op := newCTCLossOp(logProbs.Dtype(), targets.Shape().Dims(), reduction)

	return ApplyOp(op, logProbs, targets, inputLenghts, targetLenghts)
}

type ctcLossOp struct {
	dtype      tensor.Dtype
	targetDims int
	reduction  Reduction
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
	return tensor.Shape{1}, nil
}

func (op *ctcLossOp) Type() hm.Type {
	a := hm.TypeVariable('a')
	b := makeTensorType(op.targetDims, tensor.Int)
	c := makeTensorType(1, tensor.Int)

	d := makeTensorType(1, op.dtype)

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

// func (op *ctcLossOp) UsePreallocDo(prealloc Value, inputs ...Value) (Value, error) {
// }

func (op *ctcLossOp) Do(inputs ...Value) (retVal Value, err error) {
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

	targetLenghtsT := inputs[3].(*tensor.Dense)
	if targetLenghtsT.Dtype() != tensor.Int {
		return nil, fmt.Errorf("invalid type %v for inputLenghts. it should be Int", targetLenghtsT.Dtype())
	}

	targets := targetsT.Ints()
	targetLengths := targetLenghtsT.Ints()
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

	log.Printf("target stride: %v max target lenght: %v num labels: %v ", targetStride, maxTargetLength, numLabels)

	maxInputLenght := logProbsT.Shape()[0]
	for i := 0; i < batchSize; i++ {
		if inputLengths[i] > maxInputLenght {
			return nil, fmt.Errorf("expected inputLenghts to have value at most %v, but got %v", maxInputLenght, inputLengths[i])
		}
	}

	var negInf float64
	// if logProbsT.Dtype() == Float32 {
	// 	negInf = math32.Inf(-1)
	// } else {
	negInf = math.Inf(-1)
	// }

	logAlphaWidth := 2*maxTargetLength + 1
	logAlpha := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize, logProbsT.Shape()[0], logAlphaWidth),
	)

	logAlphaSpatialDim := tensor.Shape(logAlpha.Shape()[1:]).TotalSize()

	logAlphaView, err := logAlpha.Narrow(1, 0, 1)
	if err != nil {
		return nil, err
	}

	if err := logAlphaView.Memset(negInf); err != nil {
		return nil, err
	}

	negLogLikelihood := tensor.New(
		tensor.Of(logProbsT.Dtype()),
		tensor.WithShape(batchSize),
	)

	lpp, err := tensor.Transpose(logProbsT, 1, 0, 2)
	if err != nil {
		return nil, err
	}

	log.Printf("log alpha: %v", logAlpha)
	log.Printf("lp: %v", logProbsT)
	log.Printf("lpp: %v", lpp)

	logAlphaA := logAlpha.Float64s()
	lppA := lpp.(*tensor.Dense).Float64s()
	negLogLikelihoodA := negLogLikelihood.Float64s()

	_ = logAlphaA
	_ = lppA

	// this can be paralellized
	for b := 0; b < batchSize; b++ {
		inputLength := inputLengths[b]
		targetLength := targetLengths[b]
		targetWidth := 2*targetLength + 1
		targetsOffset := targetBatchOffsets[b]

		_ = inputLength
		_ = targetLength

		initialIndex := b * spatialDim
		finalIndex := (b + 1) * spatialDim
		lppSection := lppA[initialIndex:finalIndex]

		initialLogAlphaIndex := b * logAlphaSpatialDim
		finalLogAlphaIndex := (b + 1) * logAlphaSpatialDim
		logAlphaSection := logAlphaA[initialLogAlphaIndex:finalLogAlphaIndex]

		_ = lppSection
		_ = logAlphaSection

		logAlphaSection[0] = lppSection[0]

		if targetLength > 0 {
			logAlphaSection[1] = lppSection[op.getPrimeTarget(targets, targetsOffset, targetStride, 1)]
		}

		log.Printf("------- input lenght: %v spatialDim: %v alpha sptialdim: %v max target l: %v target l: %v s: %v", inputLength, spatialDim, logAlphaSpatialDim, maxTargetLength, targetLength, 2*targetLength+1)

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

	log.Printf("log alpha:\n%v", logAlpha)
	log.Printf("neg log likelihood:\n%v", negLogLikelihood)
	log.Printf("target lenghts: %v", targetLengths)

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

	log.Printf("loss: %v", loss)

	return tensor.New(
		tensor.WithShape(1),
		tensor.WithBacking([]float64{loss}),
	), nil
}
