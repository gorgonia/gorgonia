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

		inputLengthsInit  InitWFn
		inputLengthsShape tensor.Shape

		targetLengthsInit  InitWFn
		targetLengthsShape tensor.Shape

		expectedOutput    tensor.Tensor
		expectedInputGrad tensor.Tensor
	}{
		{
			Dtype:              Float64,
			reduction:          ReductionMean,
			logProbsInit:       RangedFromWithStep(0.0, 0.01),
			logProbsShape:      tensor.Shape{4, 4, 4},
			targetsInit:        RangedFromWithStep(2, 0),
			targetsShape:       tensor.Shape{4, 4},
			inputLengthsInit:   RangedFromWithStep(4, 0),
			inputLengthsShape:  tensor.Shape{4},
			targetLengthsInit:  RangedFromWithStep(2, 0),
			targetLengthsShape: tensor.Shape{4},
			expectedOutput: tensor.New(
				tensor.WithShape(),
				tensor.WithBacking([]float64{-1.428742987863855}),
			),
			expectedInputGrad: tensor.New(
				tensor.WithShape(4, 4, 4),
				tensor.WithBacking([]float64{0.8016031528676627, 1.010050167084168, 0.21859818715909318, 1.030454533953517, 0.842413927060051, 1.0512710963760241, 0.2602333936776974, 1.0725081812542165, 0.8848902205426215, 1.0941742837052104, 0.30356776520798545, 1.1162780704588713, 0.9291000044470383, 1.1388283833246218, 0.3486706459895643, 1.161834242728283, 0.5743124474256416, 1.1853048513203654, 0.7964157866879789, 1.2092495976572515, 0.6222043345940014, 1.2336780599567434, 0.8452751541535497, 1.258600009929478, 0.6720507267552366, 1.2840254166877416, 0.8961285102319406, 1.3099644507332475, 0.7239313887712684, 1.3364274880254723, 0.9490572311421717, 1.3634251141321778, 0.7779293407697887, 1.3909681284637805, 1.0041460141297627, 1.4190675485932576, 0.8341309909941721, 1.4477346146633248, 1.0614830130003936, 1.476980793882643, 0.8926262740751023, 1.506817785112854, 1.121159979184803, 1.5372575235482817, 0.9535087949451677, 1.5683121854901692, 1.1832724085606505, 1.5999941932173607, 1.4176775550605565, 1.6323162199553793, 0.8471181178324659, 1.6652911949458866, 1.4836308025665494, 1.698932308618551, 0.9144037093171966, 1.7332530178673957, 1.5522756531637645, 1.7682670514337357, 0.9844352778824115, 1.8039884153978574, 1.6237219532581721, 1.840431398781638, 1.0573248889786804, 1.8776105792643438}),
			),
		},
		{
			Dtype:              Float32,
			reduction:          ReductionSum,
			logProbsInit:       RangedFromWithStep(0.0, 0.01),
			logProbsShape:      tensor.Shape{4, 4, 4},
			targetsInit:        RangedFromWithStep(2, 0),
			targetsShape:       tensor.Shape{4, 4},
			inputLengthsInit:   RangedFromWithStep(4, 0),
			inputLengthsShape:  tensor.Shape{4},
			targetLengthsInit:  RangedFromWithStep(2, 0),
			targetLengthsShape: tensor.Shape{4},
			expectedOutput: tensor.New(
				tensor.WithShape(),
				tensor.WithBacking([]float32{-11.429942}),
			),
			expectedInputGrad: tensor.New(
				tensor.WithShape(4, 4, 4),
				tensor.WithBacking([]float32{0.8016031, 1.0100502, 0.21859795, 1.0304545, 0.8424139, 1.0512711, 0.26023316, 1.0725082, 0.88489014, 1.0941743, 0.303568, 1.116278, 0.9291, 1.1388284, 0.3486709, 1.1618342, 0.57431227, 1.1853049, 0.79641575, 1.2092496, 0.6222042, 1.2336781, 0.84527516, 1.2586, 0.6720507, 1.2840254, 0.8961285, 1.3099644, 0.7239313, 1.3364275, 0.94905716, 1.363425, 0.7779292, 1.3909681, 1.0041459, 1.4190674, 0.83413076, 1.4477345, 1.0614828, 1.4769807, 0.89262605, 1.5068176, 1.1211598, 1.5372573, 0.9535085, 1.5683119, 1.1832721, 1.599994, 1.4176772, 1.6323159, 0.8471178, 1.6652908, 1.4836304, 1.6989319, 0.9144034, 1.7332526, 1.5522753, 1.7682667, 0.98443496, 1.803988, 1.6237215, 1.8404309, 1.0573245, 1.87761}),
			),
		},
		{
			Dtype:              Float64,
			reduction:          ReductionSum,
			logProbsInit:       RangedFromWithStep(0.0, 0.01),
			logProbsShape:      tensor.Shape{4, 3, 5},
			targetsInit:        RangedFromWithStep(2, 0),
			targetsShape:       tensor.Shape{3, 4},
			inputLengthsInit:   RangedFromWithStep(4, 0),
			inputLengthsShape:  tensor.Shape{3},
			targetLengthsInit:  RangedFromWithStep(2, 0),
			targetLengthsShape: tensor.Shape{3},
			expectedOutput: tensor.New(
				tensor.WithShape(),
				tensor.WithBacking([]float64{-8.27245792718313}),
			),
			expectedInputGrad: tensor.New(
				tensor.WithShape(4, 3, 5),
				tensor.WithBacking([]float64{0.8016031528676626, 1.010050167084168, 0.21859818715909318, 1.030454533953517, 1.0408107741923882, 0.8528742492436868, 1.0618365465453596, 0.27090502838655417, 1.0832870676749586, 1.0941742837052104, 0.9067740709433104, 1.1162780704588713, 0.32589369871171314, 1.1388283833246218, 1.1502737988572274, 0.5626358191621145, 1.1735108709918103, 0.784503274886534, 1.1972173631218102, 1.2092495976572515, 0.6222043345940012, 1.2336780599567434, 0.8452751541535495, 1.258600009929478, 1.2712491503214047, 0.684826993121573, 1.2969300866657718, 0.9091628742994162, 1.323129812337437, 1.3364274880254723, 0.7506603840098345, 1.3634251141321778, 0.9763261879021259, 1.3909681284637805, 1.404947590563594, 0.8198691250270892, 1.4333294145603404, 1.0469330382294935, 1.462284589434225, 1.476980793882643, 0.8926262740751021, 1.506817785112854, 1.1211599791848028, 1.5372575235482817, 1.5527072185113364, 1.3699153383578317, 1.584073984994482, 0.7983910403496982, 1.6160744021928939, 1.6323162199553793, 1.4503244235677912, 1.6652911949458866, 0.8804244968312241, 1.698932308618551, 1.716006862184859, 1.5348561707350583, 1.7506725002961017, 0.9666638985660732, 1.7860384307500738, 1.8039884153978574}),
			),
		},
	}

	for i, tC := range testCases {
		t.Run(fmt.Sprintf("Example #%v %v (%v)", i+1, tC.Dtype, tC.logProbsShape), func(t *testing.T) {
			ac := require.New(t)

			g := NewGraph()
			logProbs := NewTensor(g, tC.Dtype, tC.logProbsShape.Dims(), WithShape(tC.logProbsShape...), WithInit(tC.logProbsInit), WithName("logProbs"))
			targets := NewTensor(g, tensor.Int, tC.targetsShape.Dims(), WithShape(tC.targetsShape...), WithInit(tC.targetsInit), WithName("targets"))
			inputLengths := NewTensor(g, tensor.Int, tC.inputLengthsShape.Dims(), WithShape(tC.inputLengthsShape...), WithInit(tC.inputLengthsInit), WithName("inputLengths"))
			targetLengths := NewTensor(g, tensor.Int, tC.targetLengthsShape.Dims(), WithShape(tC.targetLengthsShape...), WithInit(tC.targetLengthsInit), WithName("targetLengths"))

			val, err := CTCLoss(logProbs, targets, inputLengths, targetLengths, tC.reduction)
			ac.NoError(err)

			_, err = Grad(val, logProbs)
			ac.NoError(err)

			vm := NewTapeMachine(g)
			ac.NoError(vm.RunAll())

			ac.Equal(tC.expectedOutput.Shape(), val.Shape())
			ac.InDelta(tC.expectedOutput.Data(), val.Value().Data(), 1e-5, "actual: %#v", val.Value().Data())

			t.Logf("dx: %#v", logProbs.Deriv().Value().Data())

			ac.Equal(tC.expectedInputGrad.Shape(), logProbs.Deriv().Shape())
			ac.InDeltaSlice(tC.expectedInputGrad.Data(), logProbs.Deriv().Value().Data(), 1e-5, "actual: %#v", logProbs.Deriv().Value().Data())
		})
	}
}
