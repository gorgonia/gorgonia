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
	if !t.IsMatrix() {
		// error
		err = errors.Errorf(dimMismatch, 2, t.Dims())
		return
	}

	var mat *mat64.Dense
	var svd mat64.SVD
	var ok bool

	if mat, err = ToMat64(t, false); err != nil {
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
		err = errors.Errorf("SVD requires computation of `u` and `v` matrices if `full` was specified")
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

	f64backing := svd.Values(nil)
	backing := fromFloat64s(f64backing, t.t)
	s = New(WithBacking(backing))
	if uv {
		um.UFromSVD(&svd)
		vm.VFromSVD(&svd)

		u = FromMat64(&um, t.t, false)
		v = FromMat64(&vm, t.t, false)
	}

	return
}
