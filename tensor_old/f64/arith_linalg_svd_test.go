package tensorf64

import (
	"errors"
	"fmt"
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// tests for SVD adapted from Gonum's SVD tests.
// Gonum's licence is listed at https://github.com/gonum/license

var svdtestsThin = []struct {
	data  []float64
	shape types.Shape

	correctSData  []float64
	correctSShape types.Shape

	correctUData  []float64
	correctUShape types.Shape

	correctVData  []float64
	correctVShape types.Shape
}{
	{
		[]float64{2, 4, 1, 3, 0, 0, 0, 0}, types.Shape{4, 2},
		[]float64{5.464985704219041, 0.365966190626258}, types.Shape{2},
		[]float64{-0.8174155604703632, -0.5760484367663209, -0.5760484367663209, 0.8174155604703633, 0, 0, 0, 0}, types.Shape{4, 2},
		[]float64{-0.4045535848337571, -0.9145142956773044, -0.9145142956773044, 0.4045535848337571}, types.Shape{2, 2},
	},

	{
		[]float64{1, 1, 0, 1, 0, 0, 0, 0, 0, 11, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 12, 2, 1, 1, 0, 0, 0, 0, 0, 0, 1, 13, 3}, types.Shape{3, 11},
		[]float64{21.259500881097434, 1.5415021616856566, 1.2873979074613628}, types.Shape{3},
		[]float64{-0.5224167862273765, 0.7864430360363114, 0.3295270133658976, -0.5739526766688285, -0.03852203026050301, -0.8179818935216693, -0.6306021141833781, -0.6164603833618163, 0.4715056408282468}, types.Shape{3, 3},
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
			-0.16755466189153964, -0.7395268089170608, 0.08395240366704032}, types.Shape{11, 3},
	},
}

var svdtestsFull = []types.Shape{
	// {5, 5},
	{5, 3},
	// {3, 5},
	// {150, 150},
	// {200, 150},
	// {150, 200},
}

// calculate corrects
func calcSigma(s, T *Tensor, shape types.Shape) (sigma *Tensor, err error) {
	sigma = NewTensor(WithShape(shape...))
	for i := 0; i < types.MinInt(shape[0], shape[1]); i++ {
		var idx int
		if idx, err = types.Ltoi(sigma.Shape(), sigma.Strides(), i, i); err != nil {
			return
		}
		sigma.data[idx] = s.data[i]
	}

	return
}

// test svd by doing the SVD, then calculating the corrects
func testSVD(T, T2, s, u, v *Tensor, t string, i int) (err error) {
	var sigma, reconstructed *Tensor

	if !sliceApprox(T2.data, T.data, closeenough) {
		return errors.New(fmt.Sprintf("A call to SVD modified the underlying data! %s Test %d", t, i))

	}

	shape := T2.Shape()
	if t == "thin" {
		shape = types.Shape{types.MinInt(shape[0], shape[1]), types.MinInt(shape[0], shape[1])}
	}

	if sigma, err = calcSigma(s, T, shape); err != nil {
		return
	}
	v.T()

	if reconstructed, err = u.MatMul(sigma, types.UseSafe()); err != nil {
		return
	}
	if reconstructed, err = reconstructed.MatMul(v, types.UseSafe()); err != nil {
		return
	}

	if !sliceApprox(T2.data, reconstructed.data, closeenough) {
		return errors.New(fmt.Sprintf("Expected reconstructed to be %v. Got %v instead", T2.data, reconstructed.data))
	}
	return nil
}

func TestT_SVD(t *testing.T) {
	var T, T2, s, u, v *Tensor
	var err error

	// gonum specific thin special cases
	for i, stts := range svdtestsThin {
		T = NewTensor(WithShape(stts.shape...), WithBacking(stts.data))
		T2 = T.Clone()

		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if !sliceApprox(T2.data, T.data, closeenough) {
			t.Errorf("A call to SVD modified the underlying data! Thin Test %d", i)
			continue
		}

		if !sliceApprox(stts.correctSData, s.data, closeenough) {
			t.Errorf("Expected %v. Got %v instead", stts.correctSData, s.data)
		}

		if !sliceApprox(stts.correctUData, u.data, closeenough) {
			t.Errorf("Expected %v. Got %v instead", stts.correctUData, u.data)
		}

		if !sliceApprox(stts.correctVData, v.data, closeenough) {
			t.Errorf("Expected %v. Got %v instead", stts.correctVData, v.data)
		}
	}

	// standard tests
	for i, stfs := range svdtestsFull {
		T = NewTensor(WithShape(stfs...), WithBacking(RandomFloat64(stfs.TotalSize())))
		T2 = T.Clone()

		// full
		if s, u, v, err = T.SVD(true, true); err != nil {
			t.Error(err)
			continue
		}

		if err = testSVD(T, T2, s, u, v, "full", i); err != nil {
			t.Error(err)
		}

		// thin
		if s, u, v, err = T.SVD(true, false); err != nil {
			t.Error(err)
			continue
		}

		if err = testSVD(T, T2, s, u, v, "thin", i); err != nil {
			t.Error(err)
		}

		// none
		if s, u, v, err = T.SVD(false, false); err != nil {
			t.Error(err)
			continue
		}

		var svd mat64.SVD
		var mat *mat64.Dense
		if mat, err = ToMat64(T, true); err != nil {
			t.Error(err)
			continue
		}

		if !svd.Factorize(mat, matrix.SVDFull) {
			t.Errorf("Unable to factorise %v", mat)
			continue
		}

		if !sliceApprox(s.data, svd.Values(nil), closeenough) {
			t.Errorf("Singular value mismatch between Full and None decomposition. Expected %v. Got %v instead", svd.Values(nil), s.data)
		}
	}

	// illogical shit
	T = NewTensor(WithShape(2, 2))
	if _, _, _, err = T.SVD(false, true); err == nil {
		t.Errorf("Expected an error!")
	}

	// if you do this, it is bad and you should feel bad
	T = NewTensor(WithShape(2, 3, 4))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor > 2 dimensions")
	}

	T = NewTensor(WithShape(2))
	if _, _, _, err = T.SVD(true, true); err == nil {
		t.Errorf("Expecetd an error: cannot SVD() a Tensor < 2 dimensions")
	}
}
