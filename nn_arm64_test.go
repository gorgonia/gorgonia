//go:build arm64
// +build arm64

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
		ExpectedInputGrad:   []float32{0, 0, 0, 0, 0, 0},
		ExpectedScaleGrad:   []float32{-4.7683716e-07, -9.536743e-07},
		ExpectedEvalResult:  []float32{0.34658915, 0.7622689, 1.1779486, 0.37499714, 0.8247664, 1.2745357, 0.40340507, 0.8872639, 1.3711226},
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
		ExpectedInputGrad:   []float32{0, 0, 0, 0, 0, 0},
		ExpectedBiasGrad:    []float32{0.90000004, 1.2},
		ExpectedScaleGrad:   []float32{9.5399075e-09, 1.7419076e-08},
		ExpectedEvalResult:  []float32{-0.09118232, -0.18817295, -0.28516355, 0.32296348, 0.7097901, 1.0966166, 0.7371093, 1.6077532, 2.478397},
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
		ExpectedScaleGrad:   []float32{-1.013279e-06, 6.556511e-07, -1.0728836e-06, 1.2516975e-06},
		ExpectedEvalResult:  []float32{1.9384567, 4.5192976, 7.100138, 2.2775328, 5.310475, 8.343417, 2.6166089, 6.101653, 9.586697},
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
		ExpectedInputGrad:   []float64{0, -1.6917633035535369e-15, 0, 3.3835266071070737e-15, 0, -1.6917633035535369e-15, 0, 3.3835266071070737e-15, 0, -1.6917633035535369e-15, 0, 3.3835266071070737e-15},
		ExpectedBiasGrad:    []float64{1.4999999999999998, 1.7999999999999998, 2.0999999999999996, 2.3999999999999995},
		ExpectedScaleGrad:   []float64{6.661338147750939e-16, -3.1086244689504383e-15, -3.3306690738754696e-16, 2.7755575615628914e-15},
		ExpectedEvalResult:  []float64{1.9384562496620426, 4.51929706554716, 7.100137881432275, 2.2775327794466387, 5.310475635044551, 8.343418490642462, 2.616609309231235, 6.101654204541942, 9.586699099852648},
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
		ExpectedScaleGrad:   []float64{-5.551115123125783e-16, -1.1102230246251565e-15},
		ExpectedInputGrad:   []float64{0, 0, 0, 0, 0, 0},
		ExpectedEvalResult:  []float64{0.34658910799275755, 0.7622687736426385, 1.1779484392925195, 0.3749971006712683, 0.8247663575353621, 1.274535614399456, 0.40340509334977903, 0.8872639414280857, 1.3711227895063927},
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
		ExpectedScaleGrad:   []float32{-9.536743e-07, -8.940697e-08},
		ExpectedEvalResult:  []float32{0.052173994, 0.052173994, 0.052173994, 0.06482227, 0.06482227, 0.06482227, 0.07747054, 0.07747054, 0.07747054},
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
		ExpectedInputGrad:   []float64{-0.005966950859558195, -0.000942150342331545, 0.004082650174895101, 0.009107450692121748, -0.01193390171911636, -0.0018843006846630643, 0.008165300349790234, 0.018214901384243518, -0.009107450692121745, -0.004082650174895094, 0.000942150342331552, 0.005966950859558199, -0.01821490138424353, -0.008165300349790234, 0.0018843006846630621, 0.011933901719116347},
		ExpectedMean:        []float64{9.990000000000002, 17.189999999999998},
		ExpectedBiasGrad:    []float64{7.799999999999999, 12.599999999999998},
		ExpectedScaleGrad:   []float64{0.3611575330864715, 0.36115753308647924},
		ExpectedEvalResult:  []float64{-0.07605312753822552, -0.8307712234629877, 10.514207017908427, 26.15731108267525},
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
		ExpectedEvalResult:  []float32{-0.07605312753822552, -0.8307712234629877, 10.514207017908427, 26.15731108267525},
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
		ExpectedScaleGrad:   []float64{0.2457695679084111, 0.24576956790841464, 0.24576956790841997},
		ExpectedInputGrad:   []float64{-0.004601993984850206, -0.0010391599719692432, 0.0025236740409117155, 0.006086508053792683, -0.009203987969700385, -0.0020783199439384733, 0.005047348081823457, 0.012173016107585387, -0.01380598195455052, -0.0031174799159076573, 0.007571022122735208, 0.018259524161378108, -0.006086508053792679, -0.0025236740409117155, 0.0010391599719692432, 0.004601993984850211, -0.012173016107585387, -0.005047348081823475, 0.0020783199439384542, 0.009203987969700385, -0.018259524161378073, -0.007571022122735209, 0.003117479915907656, 0.013805981954550557},
		ExpectedEvalResult:  []float64{-0.8490283200965322, -3.292689453330229, 31.942343213755038, 79.73227251535778},
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
		ExpectedEvalResult:  []float32{-0.8490283200965322, -3.292689453330229, 31.942343213755038, 79.73227251535778},
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
		ExpectedEvalResult:  []float32{0.15490656, 0.39957356, 0.16663405, 0.4294255, 0.17836154, 0.45927745},
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
		ExpectedEvalResult:  []float64{0.18238597329890016, 0.431568249406907, 0.19457929954319464, 0.46171127151678276, 0.20677262578748914, 0.4918542936266586},
	},
}
