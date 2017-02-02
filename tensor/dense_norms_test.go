package tensor

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

var normtests = []NormOrder{
	FrobeniusNorm(),
	NuclearNorm(),
	InfNorm(),
	NegInfNorm(),
	Norm(0),
	Norm(1),
	Norm(-1),
	Norm(2),
	Norm(-2),
}

func testNormVal(T *Dense, ord NormOrder, want float64) error {
	retVal, err := T.Norm(ord)
	if err != nil {
		return err
	}

	if !retVal.IsScalar() {
		return errors.New("Expected Scalar")
	}

	got := retVal.ScalarValue().(float64)
	if !closef64(want, got) && !(math.IsNaN(want) && alikef64(want, got)) {
		return errors.New(fmt.Sprintf("Norm %v, Backing %v: Want %f, got %f instead", ord, T.Data(), want, got))
	}
	return nil
}

func TestTensor_Norm(t *testing.T) {
	var T *Dense
	var err error
	var backing, backing1, backing2 []float64
	var corrects map[NormOrder]float64
	var wrongs []NormOrder

	// empty
	backing = make([]float64, 0)
	T = New(WithBacking(backing))
	//TODO

	// vecktor
	backing = []float64{1, 2, 3, 4}
	backing1 = []float64{-1, -2, -3, -4}
	backing2 = []float64{-1, 2, -3, 4}

	corrects = map[NormOrder]float64{
		UnorderedNorm(): math.Pow(30, 0.5),               // Unordered
		FrobeniusNorm(): math.NaN(),                      // Frobenius
		NuclearNorm():   math.NaN(),                      // Nuclear
		InfNorm():       4,                               // Inf
		NegInfNorm():    1,                               // -Inf
		Norm(0):         4,                               // 0
		Norm(1):         10,                              // 1
		Norm(-1):        12.0 / 25.0,                     // -1
		Norm(2):         math.Pow(30, 0.5),               // 2
		Norm(-2):        math.Pow((205.0 / 144.0), -0.5), // -2
	}

	backings := [][]float64{backing, backing1, backing2}
	for ord, want := range corrects {
		for _, b := range backings {
			T = New(WithShape(len(backing)), WithBacking(b))
			if err = testNormVal(T, ord, want); err != nil {
				t.Error(err)
			}
		}
	}

	// 2x2 mat
	backing = []float64{1, 3, 5, 7}
	corrects = map[NormOrder]float64{
		UnorderedNorm(): math.Pow(84, 0.5),   // Unordered
		FrobeniusNorm(): math.Pow(84, 0.5),   // Frobenius
		NuclearNorm():   10,                  // Nuclear
		InfNorm():       12,                  // Inf
		NegInfNorm():    4,                   // -Inf
		Norm(1):         10,                  // 1
		Norm(-1):        6,                   // -1
		Norm(2):         9.1231056256176615,  // 2
		Norm(-2):        0.87689437438234041, // -2
	}

	T = New(WithShape(2, 2), WithBacking(backing))
	for ord, want := range corrects {
		if err = testNormVal(T, ord, want); err != nil {
			t.Error(err)
		}
	}

	// impossible values
	wrongs = []NormOrder{
		Norm(-3),
		Norm(0),
	}
	for _, ord := range wrongs {
		if err = testNormVal(T, ord, math.NaN()); err == nil {
			t.Errorf("Expected an error when finding norm of order %v", ord)
		}
	}

	// 3x3 mat
	// this test is added because the 2x2 example happens to have equal nuclear norm and induced 1-norm.
	// the 1/10 scaling factor accomodates the absolute tolerance used.
	backing = []float64{0.1, 0.2, 0.3, 0.6, 0, 0.5, 0.3, 0.2, 0.1}
	corrects = map[NormOrder]float64{
		FrobeniusNorm(): (1.0 / 10.0) * math.Pow(89, 0.5),
		NuclearNorm():   1.3366836911774836,
		InfNorm():       1.1,
		NegInfNorm():    0.6,
		Norm(1):         1,
		Norm(-1):        0.4,
		Norm(2):         0.88722940323461277,
		Norm(-2):        0.19456584790481812,
	}

	T = New(WithShape(3, 3), WithBacking(backing))
	for ord, want := range corrects {
		if err = testNormVal(T, ord, want); err != nil {
			t.Error(err)
		}
	}
}

func TestTensor_Norm_Axis(t *testing.T) {
	assert := assert.New(t)
	var T, sliced, expected, retVal *Dense
	var err error
	var backing []float64
	var ords []NormOrder

	t.Log("Vector Norm Tests: compare the use of axis with computing of each row or column separately")
	ords = []NormOrder{
		UnorderedNorm(),
		InfNorm(),
		NegInfNorm(),
		Norm(-1),
		Norm(0),
		Norm(1),
		Norm(2),
		Norm(3),
	}

	backing = []float64{1, 2, 3, 4, 5, 6}
	T = New(WithShape(2, 3), WithBacking(backing))

	for _, ord := range ords {
		var expecteds []*Dense
		for k := 0; k < T.Shape()[1]; k++ {
			sliced, _ = T.Slice(nil, ss(k))
			sliced = sliced.Materialize().(*Dense)
			expected, _ = sliced.Norm(ord)
			expecteds = append(expecteds, expected)
		}

		if retVal, err = T.Norm(ord, 0); err != nil {
			t.Error(err)
			continue
		}

		assert.Equal(len(expecteds), retVal.Shape()[0])
		for i, e := range expecteds {
			sliced, _ = retVal.Slice(ss(i))
			sliced = sliced.Materialize().(*Dense)
			if !allClose(e.Data(), sliced.Data()) {
				t.Errorf("Axis = 0; Ord = %v; Expected %v. Got %v instead. ret %v, i: %d", ord, e.Data(), sliced.Data(), retVal, i)
			}
		}

		// reset and do axis = 1

		expecteds = expecteds[:0]
		for k := 0; k < T.Shape()[0]; k++ {
			sliced, _ = T.Slice(ss(k))
			expected, _ = sliced.Norm(ord)
			expecteds = append(expecteds, expected)
		}
		if retVal, err = T.Norm(ord, 1); err != nil {
			t.Error(err)
			continue
		}

		assert.Equal(len(expecteds), retVal.Shape()[0])
		for i, e := range expecteds {
			sliced, _ = retVal.Slice(ss(i))
			sliced = sliced.Materialize().(*Dense)
			if !allClose(e.Data(), sliced.Data()) {
				t.Errorf("Axis = 1; Ord = %v; Expected %v. Got %v instead", ord, e.Data(), sliced.Data())
			}
		}
	}

	t.Log("Matrix Norms")

	ords = []NormOrder{
		UnorderedNorm(),
		FrobeniusNorm(),
		InfNorm(),
		NegInfNorm(),
		Norm(-2),
		Norm(-1),
		Norm(1),
		Norm(2),
	}

	axeses := [][]int{
		{0, 0},
		{0, 1},
		{0, 2},
		{1, 0},
		{1, 1},
		{1, 2},
		{2, 0},
		{2, 1},
		{2, 2},
	}

	backing = Range(Float64, 1, 25).([]float64)
	T = New(WithShape(2, 3, 4), WithBacking(backing))

	dims := T.Dims()
	for _, ord := range ords {
		for _, axes := range axeses {
			rowAxis := axes[0]
			colAxis := axes[1]

			if rowAxis < 0 {
				rowAxis += dims
			}
			if colAxis < 0 {
				colAxis += dims
			}

			if rowAxis == colAxis {

			} else {
				kthIndex := dims - (rowAxis + colAxis)
				var expecteds []*Dense

				for k := 0; k < T.Shape()[kthIndex]; k++ {
					var slices []Slice
					for s := 0; s < kthIndex; s++ {
						slices = append(slices, nil)
					}
					slices = append(slices, ss(k))
					sliced, _ = T.Slice(slices...)
					if rowAxis > colAxis {
						sliced.T()
					}
					sliced = sliced.Materialize().(*Dense)
					expected, _ = sliced.Norm(ord)
					expecteds = append(expecteds, expected)
				}

				if retVal, err = T.Norm(ord, rowAxis, colAxis); err != nil {
					t.Error(err)
					continue
				}

				for i, e := range expecteds {
					sliced, _ = retVal.Slice(ss(i))
					assert.Equal(e.Data(), sliced.Data(), "ord %v, rowAxis: %v, colAxis %v", ord, rowAxis, colAxis)
				}
			}
		}
	}

}
