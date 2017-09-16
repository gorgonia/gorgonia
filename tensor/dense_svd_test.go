package tensor

import (
	"testing"

	"github.com/pkg/errors"
	"gonum.org/v1/gonum/mat"
)

// tests for SVD adapted from Gonum's SVD tests.
// Gonum's licence is listed at https://gonum.org/v1/gonum/license

var svdtestsThin = []struct {
	data  []float64
	shape Shape

	correctSData  []float64
	correctSShape Shape

	correctUData  []float64
	correctUShape Shape

	correctVData  []float64
	correctVShape Shape
}{
	{
		[]float64{2, 4, 1, 3, 0, 0, 0, 0}, Shape{4, 2},
		[]float64{5.464985704219041, 0.365966190626258}, Shape{2},
		[]float64{-0.8174155604703632, -0.5760484367663209, -0.5760484367663209, 0.8174155604703633, 0, 0, 0, 0}, Shape{4, 2},
		[]float64{-0.4045535848337571, -0.9145142956773044, -0.9145142956773044, 0.4045535848337571}, Shape{2, 2},
	},

	{
		[]float64{1, 1, 0, 1, 0, 0, 0, 0, 0, 11, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 12, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 13, 3}, Shape{3, 11},
		[]float64{21.259500881097434, 1.5415021616856566, 1.2873979074613628}, Shape{3},
		[]float64{-0.5224167862273765, 0.7864430360363114, 0.3295270133658976, -0.5739526766688285, -0.03852203026050301, -0.8179818935216693, -0.6306021141833781, -0.6164603833618163, 0.4715056408282468}, Shape{3, 3},
		[]float64{
			-0.08123293141915189, 0.08528085505260324, -0.013165501690885152,
			-0.05423546426886932, 0.1102707844980355, 0.622210623111631,
			0, 0, 0,
			-0.0245733326078166, 0.510179651760153, 0.25596360803140994,
			0, 0, 0,
			0, 0, 0,
			-0.026997467150282436, -0.024989929445430496, -0.6353761248025164,
			0, 0, 0,
			-0.029662131661052707, -0.3999088672621176, 0.3662470150802212,
			-0.9798839760830571, 0.11328174160898856, -0.047702613241813366,
			-0.16755466189153964, -0.7395268089170608, 0.08395240366704032}, Shape{11, 3},
	},
}

var svdtestsFull = []Shape{
	{5, 5},
	{5, 3},
	{3, 5},
	{150, 150},
	{200, 150},
	{150, 200},
}

// calculate corrects
func calcSigma(s, T *Dense, shape Shape) (sigma *Dense, err error) {
	sigma = New(Of(Float64), WithShape(shape...))
	for i := 0; i < MinInt(shape[0], shape[1]); i++ {
		var idx int
		if idx, err = Ltoi(sigma.Shape(), sigma.Strides(), i, i); err != nil {
			return
		}
		sigma.Float64s()[idx] = s.Float64s()[i]
	}

	return
}

// test svd by doing the SVD, then calculating the corrects
func testSVD(T, T2, s, u, v *Dense, t string, i int) (err error) {
	var sigma, reconstructed *Dense

	if !allClose(T2.Data(), T.Data(), closeenoughf64) {
		return errors.Errorf("A call to SVD modified the underlying data! %s Test %d", t, i)
	}

	shape := T2.Shape()
	if t == "thin" {
		shape = Shape{MinInt(shape[0], shape[1]), MinInt(shape[0], shape[1])}
	}

	if sigma, err = calcSigma(s, T, shape); err != nil {
		return
	}
	v.T()

	if reconstructed, err = u.MatMul(sigma, UseSafe()); err != nil {
		return
	}
	if reconstructed, err = reconstructed.MatMul(v, UseSafe()); err != nil {
		return
	}

	if !allClose(T2.Data(), reconstructed.Data(), closeenoughf64) {
		return errors.Errorf("Expected reconstructed to be %v. Got %v instead", T2.Data(), reconstructed.Data())
	}
	return nil
}

func TestDense_SVD(t *testing.T) {
	var T, T2, s, u, v *Dense
	var err error

	// gonum specific thin special cases
	for i, stts := range svdtestsThin {
		T = New(WithShape(stts.shape...), WithBacking(stts.data))
		T2 = T.Clone().(*Dense)

		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if !allClose(T2.Data(), T.Data(), closeenoughf64) {
			t.Errorf("A call to SVD modified the underlying data! Thin Test %d", i)
			continue
		}

		if !allClose(stts.correctSData, s.Data(), closeenoughf64) {
			t.Errorf("Expected s = %v. Got %v instead", stts.correctSData, s.Data())
		}

		if !allClose(stts.correctUData, u.Data(), closeenoughf64) {
			t.Errorf("Expected u = %v. Got %v instead", stts.correctUData, u.Data())
		}

		if !allClose(stts.correctVData, v.Data(), closeenoughf64) {
			t.Errorf("Expected v = %v. Got %v instead", stts.correctVData, v.Data())
		}
	}

	// standard tests
	for i, stfs := range svdtestsFull {
		T = New(WithShape(stfs...), WithBacking(Random(Float64, stfs.TotalSize())))
		T2 = T.Clone().(*Dense)

		// full
		if s, u, v, err = T.SVD(true, true); err != nil {
			t.Error(err)
			continue
		}

		if err = testSVD(T, T2, s, u, v, "full", i); err != nil {
			t.Error(err)
			continue
		}

		// thin
		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if err = testSVD(T, T2, s, u, v, "thin", i); err != nil {
			t.Error(err)
			continue
		}

		// none
		if s, u, v, err = T.SVD(false, false); err != nil {
			t.Error(err)
			continue
		}

		var svd mat.SVD
		var m *mat.Dense
		if m, err = ToMat64(T); err != nil {
			t.Error(err)
			continue
		}

		if !svd.Factorize(m, mat.SVDFull) {
			t.Errorf("Unable to factorise %v", m)
			continue
		}

		if !allClose(s.Data(), svd.Values(nil), closeenoughf64) {
			t.Errorf("Singular value mismatch between Full and None decomposition. Expected %v. Got %v instead", svd.Values(nil), s.Data())
		}
	}

	// this is illogical
	T = New(Of(Float64), WithShape(2, 2))
	if _, _, _, err = T.SVD(false, true); err == nil {
		t.Errorf("Expected an error!")
	}

	// if you do this, it is bad and you should feel bad
	T = New(Of(Float64), WithShape(2, 3, 4))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor > 2 dimensions")
	}

	T = New(Of(Float64), WithShape(2))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor < 2 dimensions")
	}
}
