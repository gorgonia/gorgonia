package gorgonia

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"
	"gorgonia.org/dawson"
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

		ExpectedTrainResult, ExpectedOutputGrad, ExpectedBiasGrad, ExpectedScaleGrad, ExpectedInputGrad interface{}
	}{
		{
			Dtype:               tensor.Float64,
			X:                   RangedFromWithStep(0.1, 2),
			XShape:              tensor.Shape{2, 20, 2, 2},
			Groups:              2,
			Channels:            20,
			ScaleInit:           RangedFromWithStep(0.3, 0.3),
			ScaleShape:          tensor.Shape{1, 20, 1, 1},
			BiasInit:            RangedFromWithStep(0.2, 0.2),
			BiasShape:           tensor.Shape{1, 20, 1, 1},
			ExpectedTrainResult: []float64{4385.900222723546, 10064.555955139396, 4385.900222723546, 10064.555955139393},
			ExpectedOutputGrad:  []float64{0.25, 0.25, 0.25, 0.25},
			ExpectedInputGrad:   []float64{0}, // FIXME
			ExpectedBiasGrad:    []float64{51.00000000000007, 55.80000000000008, 60.60000000000008, 65.40000000000009, 70.20000000000009, 75.0000000000001, 79.8000000000001, 84.60000000000008, 89.40000000000006, 94.20000000000005, 99.00000000000003, 103.80000000000001, 108.6, 113.39999999999998, 118.19999999999996, 122.99999999999994, 127.79999999999993, 132.5999999999999, 137.3999999999999, 142.19999999999987},
			ExpectedScaleGrad:   []float64{-79.3960426535744, -67.54511124603596, -52.36760260129374, -33.863516719347764, -12.03285360019801, 13.124386756155513, 41.60820434971281, 73.41859918047385, 108.55557124843864, 147.0191205536072, -154.24403049065916, -125.76021289710187, -93.94981806634075, -58.81284599837597, -20.349296693207414, 21.440829849164896, 66.55753362874096, 115.00081464552079, 166.77067289950435, 221.86710839069167},
		},
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

			var yVal, scaleVal Value
			Read(y, &yVal)
			Read(scale, &scaleVal)

			cost, _ := Mean(y)

			_ = cost

			if _, err := Grad(cost, x, scale, bias); err != nil {
				t.Fatal(err)
			}

			m := NewTapeMachine(g, BindDualValues(x, scale, bias), TraceExec(), WithInfWatch())

			err = m.RunAll()
			c.NoError(err)

			c.NoError(m.Close())

			// for visual inspection
			t.Logf("%v input:\n%v", desc, x.Value())
			// t.Logf("%v running mean: %v", desc, op.runningMean)
			// t.Logf("%v running var: %v", desc, op.runningVariance)
			t.Logf("%v output:\n%v", desc, y.Value())
			t.Logf("%v output grad:\n%v", desc, y.Deriv().Value())
			t.Logf("%v scale grad: %v", desc, scale.Deriv().Value())
			t.Logf("%v bias grad: %v", desc, bias.Deriv().Value())
			t.Logf("%v input grad:\n%v", desc, x.Deriv().Value())

			// c.True(dawson.AllClose(tC.ExpectedMean, runningMean.Data()), "Mean doesn't match:\ngot=%#v expected=%#v", op.runningMean.Data(), tC.ExpectedMean)
			// c.True(dawson.AllClose(tC.ExpectedVariance, runningVariance.Data()), "Var doesn't match:\ngot=%#v expected=%#v", op.runningVariance.Data(), tC.ExpectedVariance)
			c.True(dawson.AllClose(tC.ExpectedTrainResult, yVal.Data()), "Wrong Output\ngot=%#v\nexpected=%#v", yVal.Data(), tC.ExpectedTrainResult)

			c.True(dawson.AllClose(tC.ExpectedOutputGrad, y.Deriv().Value().Data()), "Output Grad doesn't match:\ngot=%#v expected=%#v", y.Deriv().Value().Data(), tC.ExpectedOutputGrad)
			c.True(dawson.AllClose(tC.ExpectedBiasGrad, bias.Deriv().Value().Data()), "Bias Grad doesn't match:\ngot=%#v expected=%#v", bias.Deriv().Value().Data(), tC.ExpectedBiasGrad)
			c.True(dawson.AllClose(tC.ExpectedScaleGrad, scale.Deriv().Value().Data()), "Scale Grad doens't match:\ngot=%#v expected=%#v", scale.Deriv().Value().Data(), tC.ExpectedScaleGrad)
			c.True(dawson.AllClose(tC.ExpectedInputGrad, x.Deriv().Value().Data()), "Input Grad doesn't match:\ngot=%#v expected=%#v", x.Deriv().Value().Data(), tC.ExpectedInputGrad)
		})
	}
}
