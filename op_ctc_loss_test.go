package gorgonia

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestCTCLossDo(t *testing.T) {
	testCases := []struct {
		Dtype tensor.Dtype

		reduction Reduction

		logProbsInit  InitWFn
		logProbsShape tensor.Shape

		targetsInit  InitWFn
		targetsShape tensor.Shape

		inputLenghtsInit  InitWFn
		inputLenghtsShape tensor.Shape

		targetLenghtsInit  InitWFn
		targetLenghtsShape tensor.Shape

		expectedOutput tensor.Tensor
	}{
		{
			Dtype:              Float64,
			reduction:          ReductionMean,
			logProbsInit:       RangedFromWithStep(0.0, 0.01),
			logProbsShape:      tensor.Shape{4, 4, 4},
			targetsInit:        RangedFromWithStep(2, 0),
			targetsShape:       tensor.Shape{4, 4},
			inputLenghtsInit:   RangedFromWithStep(4, 0),
			inputLenghtsShape:  tensor.Shape{4},
			targetLenghtsInit:  RangedFromWithStep(2, 0),
			targetLenghtsShape: tensor.Shape{4},
			expectedOutput: tensor.New(
				tensor.WithShape(),
				tensor.WithBacking([]float64{-1.428742987863855}),
			),
		},
		// {
		// 	Dtype:              Float32,
		// 	reduction:          ReductionSum,
		// 	logProbsInit:       RangedFromWithStep(0.0, 0.01),
		// 	logProbsShape:      tensor.Shape{4, 4, 4},
		// 	targetsInit:        RangedFromWithStep(2, 0),
		// 	targetsShape:       tensor.Shape{4, 4},
		// 	inputLenghtsInit:   RangedFromWithStep(4, 0),
		// 	inputLenghtsShape:  tensor.Shape{4},
		// 	targetLenghtsInit:  RangedFromWithStep(2, 0),
		// 	targetLenghtsShape: tensor.Shape{4},
		// 	expectedOutput: tensor.New(
		// 		tensor.WithShape(),
		// 		tensor.WithBacking([]float32{-11.429942}),
		// 	),
		// },
	}

	for i, tC := range testCases {
		t.Run(fmt.Sprintf("Example #%v %v (%v)", i+1, tC.Dtype, tC.logProbsShape), func(t *testing.T) {
			ac := require.New(t)

			g := NewGraph()
			logProbs := NewTensor(g, tC.Dtype, tC.logProbsShape.Dims(), WithShape(tC.logProbsShape...), WithInit(tC.logProbsInit), WithName("logProbs"))
			targets := NewTensor(g, tensor.Int, tC.targetsShape.Dims(), WithShape(tC.targetsShape...), WithInit(tC.targetsInit), WithName("targets"))
			inputLenghts := NewTensor(g, tensor.Int, tC.inputLenghtsShape.Dims(), WithShape(tC.inputLenghtsShape...), WithInit(tC.inputLenghtsInit), WithName("inputLenghts"))
			targetLenghts := NewTensor(g, tensor.Int, tC.targetLenghtsShape.Dims(), WithShape(tC.targetLenghtsShape...), WithInit(tC.targetLenghtsInit), WithName("targetLenghts"))

			val, err := CTCLoss(logProbs, targets, inputLenghts, targetLenghts, tC.reduction)
			ac.NoError(err)

			_, err = Grad(val, logProbs)
			ac.NoError(err)

			vm := NewTapeMachine(g)
			ac.NoError(vm.RunAll())

			ac.Equal(tC.expectedOutput.Shape(), val.Shape())
			ac.Equal(tC.expectedOutput.Data(), val.Value().Data())
		})
	}
}
