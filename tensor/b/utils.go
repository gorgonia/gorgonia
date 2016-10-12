package tensorb

import (
	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/pkg/errors"
)

// reuseCheck checks a reuse tensor, and reshapes it to be the correct one
func reuseCheck(reuse *Tensor, as *Tensor) (err error) {
	if len(reuse.data) != as.Size() {
		err = types.NewError(types.ShapeMismatch, "Reused Tensor does not have expected shape %v. Got %v instead", as.Shape(), reuse.Shape())
		return
	}
	return reuseCheckShape(reuse, as.Shape())

}

func reuseCheckShape(reuse *Tensor, s types.Shape) (err error) {
	throw := types.BorrowInts(len(s))
	copy(throw, s)

	if err = reuse.reshape(throw...); err != nil {
		err = errors.Wrapf(err, reuseReshapeErr, s, reuse.DataSize())
		return
	}

	// clean up any funny things that may be in the reuse
	if reuse.old != nil {
		types.ReturnAP(reuse.old)
		reuse.old = nil
	}

	if reuse.transposeWith != nil {
		types.ReturnInts(reuse.transposeWith)
	}

	if reuse.viewOf != nil {
		reuse.viewOf = nil
	}
	return nil
}
