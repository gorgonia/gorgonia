package tensorf64

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix"
	"github.com/gonum/matrix/mat64"
)

// SVD does the Single Value Decomposition for the *Tensor.
//
// How it works is it temporarily converts the *Tensor into a gonum/mat64 matrix, and uses Gonum's SVD function to perform the SVD.
// In the future, when gonum/lapack fully supports float32, we'll look into rewriting this
func (t *Tensor) SVD(uv, full bool) (s, u, v *Tensor, err error) {
	if !t.IsMatrix() {
		// error
		err = dimMismatchError(2, t.Opdims())
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
		err = types.NewError(types.OpError, "SVD requires computation of `u` and `v` matrices if `full` was specified")
		return
	default:
		// by default, we return only the singular values
		ok = svd.Factorize(mat, matrix.SVDNone)
	}

	if !ok {
		// error
		err = types.NewError(types.OpError, "Unable to compute SVD")
		return
	}

	// extract values
	var um, vm mat64.Dense
	s = BorrowTensor(types.MinInt(t.Shape()[0], t.Shape()[1]))
	svd.Values(s.data)
	if uv {
		um.UFromSVD(&svd)
		vm.VFromSVD(&svd)

		u = FromMat64(&um, false)
		v = FromMat64(&vm, false)
	}

	return
}
