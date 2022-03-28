//go:build !arm64
// +build !arm64

package gorgonia

import "gorgonia.org/tensor"

var bnAllCases = []struct {
	desc string

	Dtype tensor.Dtype

	X      interface{}
	XShape tensor.Shape

	ScaleInit  InitWFn
	ScaleShape tensor.Shape

	BiasInit  InitWFn
	BiasShape tensor.Shape

	ExpectedTrainResult, ExpectedOutputGrad, ExpectedBiasGrad, ExpectedScaleGrad, ExpectedInputGrad, ExpectedMean, ExpectedVariance interface{}
	ExpectedEvalResult                                                                                                              interface{}
}{
	{
		desc:                "Float32 (3,2)",
		Dtype:               tensor.Float32,
		X:                   RangedFromWithStep(0.5, 0.01),
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float32{-0.24108347, -0.53038347, -0.81968343, 0.29999986, 0.65999985, 1.0199997, 0.84108317, 1.8503832, 2.859683},
		ExpectedMean:        []float32{0.4680, 0.4770},
		ExpectedVariance:    []float32{0.10036002, 0.10036002},
		ExpectedOutputGrad:  []float32{0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111},
		ExpectedBiasGrad:    []float32{0.90000004, 1.2},
		ExpectedInputGrad:   []float32{0, 1.075037e-06, 0, 1.075037e-06, 0, 1.075037e-06},
		ExpectedScaleGrad:   []float32{3.1355107e-09, -2.6427998e-08},
		ExpectedEvalResult:  []float32{0.31553295, 0.7072125, 1.098892, 0.3428046, 0.76857376, 1.1943429, 0.37007624, 0.82993495, 1.2897936},
	},
	{
		desc:                "Float32 (3,2)",
		Dtype:               tensor.Float32,
		X:                   []float32{-0.1607, -0.3214, 0.2000, 0.4000, 0.5607, 1.1214},
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float32{-0.2511225, -0.5524657, -0.8538088, 0.3, 0.66, 1.02, 0.8511225, 1.8724657, 2.8938088},
		ExpectedMean:        []float32{0.17999999, 0.35999998},
		ExpectedVariance:    []float32{0.21709406, 0.5683762},
		ExpectedOutputGrad:  []float32{0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111},
		ExpectedInputGrad:   []float32{0, 3.0357402e-08, 0, 3.0357402e-08, 0, 3.0357402e-08},
		ExpectedBiasGrad:    []float32{0.90000004, 1.2},
		ExpectedScaleGrad:   []float32{9.5399075e-09, 1.7419076e-08},
		ExpectedEvalResult:  []float32{-0.10514938, -0.22614017, -0.34713092, 0.29286906, 0.65569556, 1.0185219, 0.69088745, 1.5375311, 2.3841748},
	},
	{
		desc:                "Float32 (3,4)",
		Dtype:               tensor.Float32,
		X:                   RangedFromWithStep(0.5, 0.01),
		XShape:              tensor.Shape{3, 4},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 4},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 4},
		ExpectedTrainResult: []float32{-1.4914193, -3.4799783, -5.468537, 1.8, 4.1999993, 6.5999985, 5.091419, 11.879976, 18.668533},
		ExpectedMean:        []float32{0.486, 0.495, 0.50400007, 0.513},
		ExpectedVariance:    []float32{0.10144002, 0.10144002, 0.10144002, 0.10144002},
		ExpectedOutputGrad:  []float32{0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111},
		ExpectedInputGrad:   []float32{0, 0, -1.6348671e-06, 0, 0, 0, -1.6348671e-06, 0, 0, 0, -1.6348671e-06, 0},
		ExpectedBiasGrad:    []float32{1.4999999, 1.8, 2.1, 2.3999999},
		ExpectedScaleGrad:   []float32{-4.7683716e-07, 0, -6.556511e-07, 0},
		ExpectedEvalResult:  []float32{1.8381538, 4.3252144, 6.8122754, 2.1725607, 5.1176443, 8.062727, 2.5069678, 5.9100733, 9.313179},
	},
	{
		desc:                "Float64 (3,4)",
		Dtype:               tensor.Float64,
		X:                   RangedFromWithStep(0.5, 0.01),
		XShape:              tensor.Shape{3, 4},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 4},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 4},
		ExpectedTrainResult: []float64{-1.491418620064566, -3.4799767801506545, -5.468534940236742, 1.8000000000000012, 4.200000000000003, 6.600000000000002, 5.091418620064569, 11.87997678015066, 18.668534940236746},
		ExpectedMean:        []float64{0.48600000000000004, 0.49500000000000005, 0.5040000000000001, 0.5130000000000001},
		ExpectedVariance:    []float64{0.10143999999999997, 0.10143999999999997, 0.10143999999999997, 0.10143999999999997},
		ExpectedOutputGrad:  []float64{0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111},
		ExpectedInputGrad:   []float64{5.075289910660611e-16, 0, 3.045173946396366e-15, 0, 5.075289910660611e-16, 0, 3.045173946396366e-15, 0, 5.075289910660611e-16, 0, 3.045173946396366e-15, 0},
		ExpectedBiasGrad:    []float64{1.4999999999999998, 1.7999999999999998, 2.0999999999999996, 2.3999999999999995},
		ExpectedScaleGrad:   []float64{8.881784197001252e-16, -2.1094237467877974e-15, 1.2212453270876722e-15, 2.886579864025407e-15},
		ExpectedEvalResult:  []float64{1.837042238419874, 4.321883054440342, 6.806723870460809, 2.1685837337807436, 5.105526588643988, 8.04246944350723, 2.5001252291416134, 5.889170122847634, 9.278215016553652},
	},
	{
		desc:                "Float64 (3,2)",
		Dtype:               tensor.Float64,
		X:                   RangedFromWithStep(0.5, 0.01),
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float64{-0.24108325083793647, -0.5303831518434603, -0.8196830528489841, 0.30000000000000004, 0.66, 1.02, 0.8410832508379366, 1.8503831518434604, 2.8596830528489843},
		ExpectedMean:        []float64{0.4680, 0.4770},
		ExpectedVariance:    []float64{0.10035999999999998, 0.10035999999999998},
		ExpectedOutputGrad:  []float64{0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111, 0.1111111111111111},
		ExpectedBiasGrad:    []float64{0.8999999999999998, 1.2},
		ExpectedScaleGrad:   []float64{0, -1.4432899320127035e-15},
		ExpectedInputGrad:   []float64{0, -2.0024102777310188e-15, 0, -2.0024102777310188e-15, 0, -2.0024102777310188e-15},
		ExpectedEvalResult:  []float64{0.3155331207656929, 0.7072127868293113, 1.0988924528929298, 0.3428047939563865, 0.7685740513435537, 1.194343308730721, 0.3700764671470802, 0.8299353158577962, 1.2897941645685123},
	},
	{
		desc:                "Float32 (3,2)",
		Dtype:               tensor.Float32,
		X:                   RangedFromWithStep(0.1, 0.001),
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           Ones(),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            Zeroes(),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float32{-1.1239058, -1.1239058, -1.1239058, -1.0457852e-06, -1.0457852e-06, -1.0457852e-06, 1.1239038, 1.1239038, 1.1239038},
		ExpectedMean:        []float32{0.0918, 0.0927},
		ExpectedVariance:    []float32{0.10000362, 0.10000362},
		ExpectedOutputGrad:  []float32{0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111, 0.11111111},
		ExpectedInputGrad:   []float32{0, 0, 0, 0, 0, 0},
		ExpectedBiasGrad:    []float32{1, 1},
		ExpectedScaleGrad:   []float32{-6.4074993e-07, 0},
		ExpectedEvalResult:  []float32{0.012807339, 0.012807339, 0.012807339, 0.025703307, 0.025703307, 0.025703307, 0.038599305, 0.038599305, 0.038599305},
	},
	{
		desc:                "Float64 (2,2,2,2)",
		Dtype:               tensor.Float64,
		X:                   RangedFromWithStep(0.1, 2),
		XShape:              tensor.Shape{2, 2, 2, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2, 1, 1},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2, 1, 1},
		ExpectedTrainResult: []float64{-1.328982312548889, -3.8900518748612214, 9.419065872104541, 23.500135434416876},
		ExpectedVariance:    []float64{71.07142857142858, 71.07142857142858},
		ExpectedOutputGrad:  []float64{0.25, 0.25, 0.25, 0.25},
		ExpectedInputGrad:   []float64{-0.005966950859558195, -0.000942150342331545, 0.004082650174895101, 0.009107450692121748, -0.011933901719116366, -0.001884300684663068, 0.008165300349790232, 0.018214901384243514, -0.009107450692121745, -0.004082650174895094, 0.000942150342331552, 0.005966950859558199, -0.018214901384243528, -0.008165300349790234, 0.0018843006846630658, 0.011933901719116352},
		ExpectedMean:        []float64{9.990000000000002, 17.189999999999998},
		ExpectedBiasGrad:    []float64{7.799999999999999, 12.599999999999998},
		ExpectedScaleGrad:   []float64{0.3611575330864715, 0.36115753308647924},
		ExpectedEvalResult:  []float64{-0.13753229022631047, -0.9806340999853157, 9.91220707244465, 24.738135347525837},
	},
	{
		desc:                "Float32 (2,2,2,2)",
		Dtype:               tensor.Float32,
		X:                   RangedFromWithStep(0.1, 2),
		XShape:              tensor.Shape{2, 2, 2, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2, 1, 1},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2, 1, 1},
		ExpectedTrainResult: []float32{-1.328982312548889, -3.8900518748612214, 9.419065872104541, 23.500135434416876},
		ExpectedVariance:    []float32{71.07142857142858, 71.07142857142858},
		ExpectedOutputGrad:  []float32{0.25, 0.25, 0.25, 0.25},
		ExpectedInputGrad:   []float32{-0.0059669508595582, -0.0009421503423315501, 0.004082650174895096, 0.009107450692121741, -0.011933901719116317, -0.0018843006846630268, 0.008165300349790263, 0.01821490138424354, -0.009107450692121745, -0.004082650174895095, 0.0009421503423315506, 0.005966950859558196, -0.01821490138424354, -0.00816530034979025, 0.0018843006846630404, 0.011933901719116315},
		ExpectedMean:        []float32{9.990000000000002, 17.189999999999998},
		ExpectedBiasGrad:    []float32{7.799999999999999, 12.599999999999998},
		ExpectedScaleGrad:   []float32{0.3611575330864715, 0.36115753308647924},
		ExpectedEvalResult:  []float32{-0.13753274, -0.9806347, 9.912203, 24.738127},
	},
	{
		desc:                "Float64 (2,3,2,2)",
		Dtype:               tensor.Float64,
		X:                   RangedFromWithStep(0.1, 2),
		XShape:              tensor.Shape{2, 3, 2, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 3, 1, 1},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 3, 1, 1},
		ExpectedTrainResult: []float64{-4.911299133806133, -13.11268793455021, 28.356069578276426, 71.11745837902049},
		ExpectedMean:        []float64{13.590000000000002, 20.79, 27.99},
		ExpectedVariance:    []float64{153.35714285714286, 153.3571428571429, 153.35714285714286},
		ExpectedOutputGrad:  []float64{0.25, 0.25, 0.25, 0.25},
		ExpectedBiasGrad:    []float64{10.199999999999998, 14.999999999999996, 19.799999999999997},
		ExpectedScaleGrad:   []float64{0.24576956790840931, 0.24576956790841376, 0.24576956790841997},
		ExpectedInputGrad:   []float64{-0.004601993984850203, -0.00103915997196924, 0.002523674040911718, 0.006086508053792685, -0.009203987969700385, -0.0020783199439384737, 0.005047348081823455, 0.012173016107585385, -0.013805981954550545, -0.003117479915907676, 0.007571022122735192, 0.018259524161378098, -0.00608650805379268, -0.002523674040911718, 0.0010391599719692401, 0.004601993984850207, -0.012173016107585385, -0.0050473480818234735, 0.0020783199439384547, 0.009203987969700383, -0.01825952416137806, -0.007571022122735192, 0.0031174799159076746, 0.013805981954550578},
		ExpectedEvalResult:  []float64{-0.963147483026699, -3.5947482385463116, 30.651455733371414, 76.57899246632469},
	},
	{
		desc:                "Float32 (2,3,2,2)",
		Dtype:               tensor.Float32,
		X:                   RangedFromWithStep(0.1, 2),
		XShape:              tensor.Shape{2, 3, 2, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 3, 1, 1},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 3, 1, 1},
		ExpectedTrainResult: []float32{-4.911299133806133, -13.11268793455021, 28.356069578276426, 71.11745837902049},
		ExpectedMean:        []float32{13.590000000000002, 20.79, 27.99},
		ExpectedVariance:    []float32{153.35714285714286, 153.3571428571429, 153.35714285714286},
		ExpectedOutputGrad:  []float32{0.25, 0.25, 0.25, 0.25},
		ExpectedBiasGrad:    []float32{10.199999999999998, 14.999999999999996, 19.799999999999997},
		ExpectedScaleGrad:   []float32{0.24577019, 0.24577145, 0.24576364},
		ExpectedInputGrad:   []float32{-0.0046019927, -0.0010391591, 0.002523677, 0.00608651, -0.009203983, -0.0020783104, 0.0050473553, 0.012173022, -0.013806061, -0.0031175415, 0.00757096, 0.018259479, -0.00608651, -0.0025236772, 0.001039159, 0.004601992, -0.012173033, -0.0050473614, 0.002078304, 0.0092039695, -0.018259495, -0.007570976, 0.003117525, 0.013806043},
		ExpectedEvalResult:  []float32{-0.96314955, -3.5947528, 30.651447, 76.57899},
	},
}

var bnstackedCases = []struct {
	desc string

	Epochs int

	Dtype tensor.Dtype

	XInit  InitWFn
	XShape tensor.Shape

	ScaleInit  InitWFn
	ScaleShape tensor.Shape

	BiasInit  InitWFn
	BiasShape tensor.Shape

	ExpectedTrainResult, ExpectedOutputGrad, ExpectedBiasGrad, ExpectedScaleGrad, ExpectedMean, ExpectedVariance interface{}
	ExpectedEvalResult                                                                                           interface{}
}{
	{
		desc:                "Example (1d Float32)",
		Dtype:               tensor.Float32,
		Epochs:              1,
		XInit:               RangedFromWithStep(float32(0.5), float32(0.01)),
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           RangedFromWithStep(float32(0.3), float32(0.3)),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            RangedFromWithStep(float32(0.2), float32(0.2)),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float32{-0.16740213, -0.33483604, 0.19999963, 0.39999926, 0.5674025, 1.1348367},
		ExpectedMean:        []float32{0.18000033, 0.36000067},
		ExpectedVariance:    []float32{0.21710846, 0.56843376},
		ExpectedOutputGrad:  []float32{0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666},
		ExpectedBiasGrad:    []float32{0.5, 0.5},
		ExpectedScaleGrad:   []float32{1.6863456e-08, -8.432093e-09},
		ExpectedEvalResult:  []float32{0.15759754, 0.40007672, 0.17100696, 0.43021968, 0.18441638, 0.4603627},
	},
	{
		desc:                "Example (1d Float64)",
		Dtype:               tensor.Float64,
		Epochs:              1,
		XInit:               RangedFromWithStep(0.5, 0.01),
		XShape:              tensor.Shape{3, 2},
		ScaleInit:           RangedFromWithStep(0.3, 0.3),
		ScaleShape:          tensor.Shape{1, 2},
		BiasInit:            RangedFromWithStep(0.2, 0.2),
		BiasShape:           tensor.Shape{1, 2},
		ExpectedTrainResult: []float64{-0.1674022853682372, -0.33483633412378944, 0.19999999999999998, 0.4, 0.5674022853682372, 1.1348363341237895},
		ExpectedMean:        []float64{0.17999999999999985, 0.36000000000000015},
		ExpectedVariance:    []float64{0.21710843373493974, 0.568433734939759},
		ExpectedOutputGrad:  []float64{0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666, 0.16666666666666666},
		ExpectedBiasGrad:    []float64{0.5, 0.5},
		ExpectedScaleGrad:   []float64{1.6863456e-08, -8.432093e-09},
		ExpectedEvalResult:  []float64{0.1823859733138815, 0.4315682494492552, 0.1945792995638166, 0.4617112715803343, 0.20677262581375164, 0.4918542937114134},
	},
}
