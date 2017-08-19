package tensor

import "github.com/pkg/errors"

// SVD does the Single Value Decomposition for the *Dense.
//
// How it works is it temporarily converts the *Dense into a gonum/mat64 matrix, and uses Gonum's SVD function to perform the SVD.
// In the future, when gonum/lapack fully supports float32, we'll look into rewriting this
func (t *Dense) SVD(uv, full bool) (s, u, v *Dense, err error) {
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}
	if svder, ok := e.(SVDer); ok {
		var sT, uT, vT Tensor
		if sT, uT, vT, err = svder.SVD(t, uv, full); err != nil {
			return nil, nil, nil, errors.Wrap(err, "Error while performing *Dense.SVD")
		}
		if s, err = assertDense(sT); err != nil {
			return nil, nil, nil, errors.Wrapf(err, "sT is not *Dense (uv %t full %t). Got %T instead", uv, full, sT)
		}
		// if not uv and not full, u can be nil
		if u, err = assertDense(uT); err != nil && !(!uv && !full) {
			return nil, nil, nil, errors.Wrapf(err, "uT is not *Dense (uv %t full %t). Got %T instead", uv, full, uT)
		}
		// if not uv and not full, v can be nil
		if v, err = assertDense(vT); err != nil && !(!uv && !full) {
			return nil, nil, nil, errors.Wrapf(err, "vT is not *Dense (uv %t full %t). Got %T instead", uv, full, vT)
		}
		return s, u, v, nil
	}
	return nil, nil, nil, errors.New("Engine does not support SVD")
}
