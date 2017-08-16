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
	e := t.Engine()
	if e == nil {
		e = StdEng{}
	}
	if svder, ok := e.(SVDer); ok {
		var sT, uT, vT Tensor
		if sT, uT, vT, er = svder.SVD(t, uv, full);err != nil {
			return nil, nil, nil, errors.Wrap(err, "Error while performing *Dense.SVD")
		}
		return sT.(*Dense), uT.(*Dense), vT.(*Dense), nil
	}
	return nil, nil, nil, errors.New("Engine does not support SVD")
}
