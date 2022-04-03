package gorgonia

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/tensor"
)

func TestGroupNorm(t *testing.T) {
	testCases := []struct {
		Dtype  tensor.Dtype
		X      interface{}
		XShape tensor.Shape

		Groups, Channels int

		ScaleInit  InitWFn
		ScaleShape tensor.Shape

		BiasInit  InitWFn
		BiasShape tensor.Shape

		ExpectedTrainResult, ExpectedCost                      interface{}
		ExpectedBiasGrad, ExpectedScaleGrad, ExpectedInputGrad interface{}
	}{
		// {
		// 	Dtype:               tensor.Float64,
		// 	X:                   RangedFromWithStep(0.1, 2),
		// 	XShape:              tensor.Shape{2, 20, 2, 2},
		// 	Groups:              2,
		// 	Channels:            20,
		// 	ScaleInit:           RangedFromWithStep(0.3, 0.3),
		// 	ScaleShape:          tensor.Shape{1, 20, 1, 1},
		// 	BiasInit:            RangedFromWithStep(0.2, 0.2),
		// 	BiasShape:           tensor.Shape{1, 20, 1, 1},
		// 	ExpectedTrainResult: []float64{4385.900222723546, 10064.555955139396, 4385.900222723546, 10064.555955139393},
		// 	ExpectedCost:        7225.228088931472,
		// 	ExpectedInputGrad:   []float64{0.1455456629614057, 0.10756983263957326, 0.06959400231774085, 0.03161817199590841, 0.08135482742065983, 0.04532816344875147, 0.009301499476843111, -0.02672516449506525, 0.03275732267930655, -0.0013201749426776932, -0.03539767256466193, -0.06947517018664628, -0.00024685126265393265, -0.03237518253471408, -0.06450351380677433, -0.09663184507883454, -0.017657694405221844, -0.047836859327357845, -0.07801602424949394, -0.10819518917163007, -0.019475206748396964, -0.04770520532060915, -0.07593520389282099, -0.10416520246503308, -0.005699388292179461, -0.03198022051446739, -0.058261052736755414, -0.08454188495904336, 0.023669760963430445, -0.0006619049089335582, -0.02499357078129756, -0.04932523665366134, 0.068632241018433, 0.04624974149599304, 0.023867241973553298, 0.0014847424511133245, 0.12918805187282817, 0.10875471870031223, 0.08832138552779653, 0.06788805235528059, 0.1712175958138913, 0.11393295134647796, 0.05664830687906508, -0.0006363375883475797, 0.10775815768778663, 0.05242267957029734, -0.0029127985471915085, -0.05824827666467991, 0.059892050361074256, 0.006505738593509225, -0.04688057317405536, -0.1002668849416204, 0.027619273833755065, -0.023817871583885708, -0.07525501700152648, -0.12669216241916725, 0.010939828105828608, -0.038548150961888794, -0.08803613002960531, -0.13752410909732182, 0.009853713177293999, -0.0376850995404987, -0.08522391225829185, -0.13276272497608366, 0.02436092904815257, -0.021228717319716317, -0.06681836368758476, -0.11240801005545364, 0.054461475718402985, 0.010820995700458802, -0.032819484317485825, -0.07645996433542956, 0.10015535318804658, 0.05846403952002577, 0.016772725852005843, -0.02491858781601497, 0.16144256145708247, 0.12170041413898636, 0.08195826682088936, 0.042216119502794136, 0.14554566296140425, 0.10756983263957176, 0.06959400231773927, 0.03161817199590722, 0.0813548274206588, 0.045328163448750125, 0.00930149947684189, -0.026725164495066345, 0.032757322679305645, -0.0013201749426783316, -0.03539767256466275, -0.06947517018664673, -0.0002468512626543351, -0.0323751825347145, -0.06450351380677466, -0.09663184507883482, -0.017657694405222024, -0.04783685932735793, -0.07801602424949383, -0.10819518917163018, -0.019475206748396978, -0.04770520532060907, -0.07593520389282071, -0.1041652024650328, -0.005699388292179197, -0.031980220514466584, -0.05826105273675486, -0.08454188495904269, 0.02366976096343132, -0.0006619049089326978, -0.024993570781296715, -0.04932523665366029, 0.06863224101843413, 0.04624974149599437, 0.02386724197355461, 0.001484742451114407, 0.12918805187282967, 0.10875471870031372, 0.08832138552779778, 0.06788805235528228, 0.17121759581388218, 0.1139329513464693, 0.05664830687905731, -0.0006363375883546851, 0.10775815768777974, 0.052422679570290676, -0.0029127985471975038, -0.058248276664684795, 0.059892050361068705, 0.00650573859350434, -0.046880573174060025, -0.10026688494162439, 0.027619273833751734, -0.023817871583888817, -0.07525501700152937, -0.12669216241916992, 0.010939828105827054, -0.03854815096188968, -0.08803613002960642, -0.13752410909732227, 0.009853713177294665, -0.03768509954049826, -0.08522391225829029, -0.13276272497608232, 0.024360929048154567, -0.021228717319713653, -0.06681836368758098, -0.11240801005545009, 0.05446147571840765, 0.010820995700464131, -0.032819484317480274, -0.07645996433542379, 0.10015535318805302, 0.05846403952003243, 0.016772725852013615, -0.024918587816006976, 0.16144256145709068, 0.12170041413899568, 0.0819582668208989, 0.042216119502803906},
		// 	ExpectedBiasGrad:    []float64{51.00000000000007, 55.80000000000008, 60.60000000000008, 65.40000000000009, 70.20000000000009, 75.0000000000001, 79.8000000000001, 84.60000000000008, 89.40000000000006, 94.20000000000005, 99.00000000000003, 103.80000000000001, 108.6, 113.39999999999998, 118.19999999999996, 122.99999999999994, 127.79999999999993, 132.5999999999999, 137.3999999999999, 142.19999999999987},
		// 	ExpectedScaleGrad:   []float64{-79.3960426535744, -67.54511124603596, -52.36760260129374, -33.863516719347764, -12.03285360019801, 13.124386756155513, 41.60820434971281, 73.41859918047385, 108.55557124843864, 147.0191205536072, -154.24403049065916, -125.76021289710187, -93.94981806634075, -58.81284599837597, -20.349296693207414, 21.440829849164896, 66.55753362874096, 115.00081464552079, 166.77067289950435, 221.86710839069167},
		// },
		{
			Dtype:               Float64,
			X:                   RangedFromWithStep(0.5, 5),
			XShape:              tensor.Shape{3, 2},
			Groups:              2,
			Channels:            2,
			ScaleInit:           RangedFromWithStep(0.3, 0.3),
			ScaleShape:          tensor.Shape{1, 2},
			BiasInit:            RangedFromWithStep(0.2, 0.2),
			BiasShape:           tensor.Shape{1, 2},
			ExpectedTrainResult: []float64{0.3, 0.66, 1.02, 0.3000000000000614, 0.6600000000001024, 1.0200000000001432, 0.3000000000000614, 0.6600000000001024, 1.0200000000001432},
			ExpectedCost:        0.6600000000000682,
			ExpectedInputGrad:   []float64{0, 0, 0, 0, 0, 0},
			ExpectedBiasGrad:    []float64{0.8999999999999998, 1.2},
			ExpectedScaleGrad:   []float64{0, 0},
		},
		// {
		// 	Dtype:               tensor.Float64,
		// 	X:                   RangedFromWithStep(0.1, 2),
		// 	XShape:              tensor.Shape{2, 20, 4, 4},
		// 	Groups:              2,
		// 	Channels:            20,
		// 	ScaleInit:           RangedFromWithStep(0.3, 0.3),
		// 	ScaleShape:          tensor.Shape{1, 20, 1, 1},
		// 	BiasInit:            RangedFromWithStep(0.2, 0.2),
		// 	BiasShape:           tensor.Shape{1, 20, 1, 1},
		// 	ExpectedTrainResult: []float64{},
		// 	ExpectedCost:        0.0,
		// 	ExpectedInputGrad:   []float64{},
		// 	ExpectedBiasGrad:    []float64{},
		// 	ExpectedScaleGrad:   []float64{},
		// },
	}

	for i, tC := range testCases {
		desc := fmt.Sprintf("Example #%d %v - %v", i+1, tC.Dtype, tC.XShape)
		t.Run(desc, func(t *testing.T) {
			rand.Seed(0)

			c := require.New(t)

			g := NewGraph()

			var initOpt NodeConsOpt

			switch v := tC.X.(type) {
			case []float32:
				initOpt = WithValue(
					tensor.New(
						tensor.Of(tensor.Float32),
						tensor.WithShape(tC.XShape...),
						tensor.WithBacking(v),
					),
				)
			case []float64:
				initOpt = WithValue(
					tensor.New(
						tensor.Of(tensor.Float32),
						tensor.WithShape(tC.XShape...),
						tensor.WithBacking(v),
					),
				)
			case InitWFn:
				initOpt = WithInit(v)
			}

			var err error

			x := NewTensor(g, tC.Dtype, tC.XShape.Dims(), WithShape(tC.XShape...), initOpt, WithName("x"))

			scale := NewTensor(g, tC.Dtype, tC.ScaleShape.Dims(), WithShape(tC.ScaleShape...), WithInit(tC.ScaleInit), WithName("scale"))
			bias := NewTensor(g, tC.Dtype, tC.BiasShape.Dims(), WithShape(tC.BiasShape...), WithInit(tC.BiasInit), WithName("bias"))

			fcWeight := NewTensor(g, tC.Dtype, 2, WithShape(tC.XShape[0], tensor.Shape(tC.XShape[1:]).TotalSize()), WithInit(tC.ScaleInit), WithName("fcWeight"))

			y, err := GroupNorm(x, scale, bias, tC.Groups, tC.Channels, 1e-5)
			c.NoError(err)

			if y.Dims() > 2 {
				y = Must(Reshape(y, fcWeight.Shape()))
			}

			wT := Must(Transpose(fcWeight, 1, 0))

			y = Must(Mul(y, wT))

			cost := Must(Mean(y))

			if _, err := Grad(cost, x, fcWeight, scale, bias); err != nil {
				t.Fatal(err)
			}

			m := NewTapeMachine(g, BindDualValues(x, fcWeight, scale, bias), TraceExec(), WithInfWatch(), WithNaNWatch())

			err = m.RunAll()
			c.NoError(err)

			c.NoError(m.Close())

			t.Logf("%v output:\n%v", desc, y.Value())
			t.Logf("%v cost:\n%v", desc, cost.Value())
			t.Logf("%v input grad:\n%v", desc, x.Deriv().Value())
			// t.Logf("%v output grad:\n%v", desc, y.Deriv().Value())
			// t.Logf("%v bias grad: %v", desc, bias.Deriv().Value())
			// t.Logf("%v scale grad: %v", desc, scale.Deriv().Value())

			c.InDeltaSlice(tC.ExpectedTrainResult, y.Value().Data(), 1e-5, "Wrong Output\ngot=%#v\nexpected=%#v", y.Value().Data(), tC.ExpectedTrainResult)
			c.InDelta(tC.ExpectedCost, cost.Value().Data(), 1e-5)

			c.InDeltaSlice(tC.ExpectedBiasGrad, bias.Deriv().Value().Data(), 1e-5, "Bias Grad doesn't match:\ngot=%#v expected=%#v", bias.Deriv().Value().Data(), tC.ExpectedBiasGrad)
			c.InDeltaSlice(tC.ExpectedScaleGrad, scale.Deriv().Value().Data(), 1e-5, "Scale Grad doens't match:\ngot=%#v expected=%#v", scale.Deriv().Value().Data(), tC.ExpectedScaleGrad)
			c.InDeltaSlice(tC.ExpectedInputGrad, x.Deriv().Value().Data(), 1e-5, "Input Grad doesn't match:\ngot=%#v expected=%#v", x.Deriv().Value().Data(), tC.ExpectedInputGrad)
		})
	}
}
