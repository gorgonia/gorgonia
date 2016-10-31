package tensorf64

import (
	"math"
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
)

var normtests = []types.NormOrder{
	types.FrobeniusNorm(),
	types.NuclearNorm(),
	types.InfNorm(),
	types.NegInfNorm(),
	types.Norm(0),
	types.Norm(1),
	types.Norm(-1),
	types.Norm(2),
	types.Norm(-2),
}

func TestT_Norm(t *testing.T) {
	var T, retVal *Tensor
	var err error
	var backing, backing1, backing2 []float64

	// empty
	backing = make([]float64, 0)
	T = NewTensor(WithBacking(backing))

	// vecktor
	backing = []float64{1, 2, 3, 4}
	backing1 = []float64{-1, -2, -3, -4}
	backing2 = []float64{-1, 2, -3, 4}

	corrects := map[types.NormOrder]float64{
		types.FrobeniusNorm(): math.Pow(30, 0.5),               // Frobenius
		types.NuclearNorm():   math.NaN(),                      // Nuclear
		types.InfNorm():       4,                               // Inf
		types.NegInfNorm():    1,                               // -Inf
		types.Norm(0):         4,                               // 0
		types.Norm(1):         10,                              // 1
		types.Norm(-1):        12.0 / 25.0,                     // -1
		types.Norm(2):         math.Pow(30, 0.5),               // 2
		types.Norm(-2):        math.Pow((205.0 / 144.0), -0.5), // -2
	}
	T = NewTensor(WithShape(len(backing)))

	backings := [][]float64{backing, backing1, backing2}
	for ord, want := range corrects {
		for i, b := range backings {
			T.data = b
			if retVal, err = T.Norm(ord, nil); err != nil {
				t.Error(err)
				continue
			}

			if !retVal.IsScalar() {
				t.Errorf("Expected Scalar. Vector, test %d, backing: %d", ord, i)
				continue
			}

			got := retVal.ScalarValue().(float64)
			if !close(want, got) && !(math.IsNaN(want) && alike(want, got)) {
				t.Errorf("Norm %v, Backing %d: Want %f, got %f instead", ord, i, want, got)
			}

			// t.Logf("%3.9f, %3.9f", retVal.data[0], corrects[i])
		}
	}
}
