package tensorf64

import (
	"errors"
	"fmt"
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

func testNormVal(T *Tensor, ord types.NormOrder, want float64) error {
	retVal, err := T.Norm(ord, nil)
	if err != nil {
		return err
	}

	if !retVal.IsScalar() {
		return errors.New("Expected Scalar")
	}

	got := retVal.ScalarValue().(float64)
	if !close(want, got) && !(math.IsNaN(want) && alike(want, got)) {
		return errors.New(fmt.Sprintf("Norm %v, Backing %v: Want %f, got %f instead", ord, T.data, want, got))
	}
	return nil
}

func TestT_Norm(t *testing.T) {
	var T *Tensor
	var err error
	var backing, backing1, backing2 []float64
	var corrects map[types.NormOrder]float64

	// empty
	backing = make([]float64, 0)
	T = NewTensor(WithBacking(backing))
	//TODO

	// vecktor
	backing = []float64{1, 2, 3, 4}
	backing1 = []float64{-1, -2, -3, -4}
	backing2 = []float64{-1, 2, -3, 4}

	corrects = map[types.NormOrder]float64{
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
		for _, b := range backings {
			T.data = b
			if err = testNormVal(T, ord, want); err != nil {
				t.Error(err)
			}
		}
	}

	// 2x2 mat
	backing = []float64{1, 3, 5, 7}
	corrects = map[types.NormOrder]float64{
		types.FrobeniusNorm(): math.Pow(84, 0.5),   // Frobenius
		types.NuclearNorm():   10,                  // Nuclear
		types.InfNorm():       12,                  // Inf
		types.NegInfNorm():    4,                   // -Inf
		types.Norm(0):         math.NaN(),          // 0
		types.Norm(1):         10,                  // 1
		types.Norm(-1):        6,                   // -1
		types.Norm(2):         9.1231056256176615,  // 2
		types.Norm(-2):        0.87689437438234041, // -2
	}

	T = NewTensor(WithShape(2, 2), WithBacking(backing))
	for ord, want := range corrects {
		if err = testNormVal(T, ord, want); err != nil {
			t.Error(err)
		}
	}

	// 3x3 mat
	// this test is added because the 2x2 example happens to have equal nuclear norm and induced 1-norm.
	// the 1/10 scaling factor accomodates the absolute tolerance used.
	backing = []float64{0.1, 0.2, 0.3, 0.6, 0, 0.5, 0.3, 0.2, 0.1}
	corrects = map[types.NormOrder]float64{
		types.FrobeniusNorm(): (1.0 / 10.0) * math.Pow(89, 0.5),
		types.NuclearNorm():   1.3366836911774836,
		types.InfNorm():       1.1,
		types.NegInfNorm():    0,
		types.Norm(1):         1,
		types.Norm(-1):        0.4,
		types.Norm(2):         0.88722940323461277,
		types.Norm(-2):        0.19456584790481812,
	}

	T = NewTensor(WithShape(3, 3), WithBacking(backing))
	for ord, want := range corrects {
		if err = testNormVal(T, ord, want); err != nil {
			t.Error(err)
		}
	}

}
