package tensor

import "github.com/pkg/errors"

/* Argmax */

// Argmax finds the index of the max value along the axis provided
func (t *Dense) Argmax(axis int) (retVal *Dense, err error) {
	e := t.e
	switch am := e.(type) {
	case denseArgmaxer:
		return am.argmaxDenseTensor(t, axis)
	case Argmaxer:
		var ret Tensor
		var ok bool
		if ret, err = am.Argmax(t, axis); err != nil {
			return nil, errors.Wrapf(err, opFail, "Argmax")
		}
		if retVal, ok = ret.(*Dense); !ok {
			return nil, errors.Errorf(extractionFail, "*Dense", ret)
		}
		return
	}
	return nil, errors.New("Engine does not suport Argmax")
}

/* Argmin */

// Argmin finds the index of the min value along the axis provided
func (t *Dense) Argmin(axis int) (retVal *Dense, err error) {
	e := t.e
	switch am := e.(type) {
	case denseArgminer:
		return am.argminDenseTensor(t, axis)
	case Argminer:
		var ret Tensor
		var ok bool
		if ret, err = am.Argmin(t, axis); err != nil {
			return nil, errors.Wrapf(err, opFail, "Argmax")
		}
		if retVal, ok = ret.(*Dense); !ok {
			return nil, errors.Errorf(extractionFail, "*Dense", ret)
		}
		return
	}
	return nil, errors.New("Engine does not suport Argmax")
}
