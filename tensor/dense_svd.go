package tensor

import (
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
	"github.com/pkg/errors"
)

// SVD does the Single Value Decomposition for the *Dense.
//
// How it works is it temporarily converts the *Dense into a gonum/mat64 matrix, and uses Gonum's SVD function to perform the SVD.
// In the future, when gonum/lapack fully supports float32, we'll look into rewriting this
func (t *Dense) SVD(uv, full bool) (s, u, v *Dense, err error) {
	if !isFloat(t.t) {
		err = errors.Errorf("Can only do SVD with floating point types")
		return
	}

	if !t.IsMatrix() {
		// error
		err = errors.Errorf(dimMismatch, 2, t.Dims())
		return
	}

	var mat *mat64.Dense
	var svd mat64.SVD
	var ok bool

	if mat, err = ToMat64(t, UseUnsafe()); err != nil {
		return
	}

	switch {
	case full && uv:
		ok = svd.Factorize(mat, matrix.SVDFull)
	case !full && uv:
		ok = svd.Factorize(mat, matrix.SVDThin)
	case full && !uv:
		// illogical state - if you specify "full", you WANT the UV matrices
		// error
		err = errors.Errorf("SVD requires computation of `u` and `v` matrices if `full` was specified.")
		return
	default:
		// by default, we return only the singular values
		ok = svd.Factorize(mat, matrix.SVDNone)
	}

	if !ok {
		// error
		err = errors.Errorf("Unable to compute SVD")
		return
	}

	// extract values
	var um, vm mat64.Dense
	s = recycledDense(Float64, Shape{MinInt(t.Shape()[0], t.Shape()[1])})
	svd.Values(s.Float64s())
	if uv {
		um.UFromSVD(&svd)
		vm.VFromSVD(&svd)

		u = FromMat64(&um, UseUnsafe(), As(t.t))
		v = FromMat64(&vm, UseUnsafe(), As(t.t))
	}

	return
}
