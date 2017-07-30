package tensor

import "github.com/pkg/errors"

func (t *Dense) Reduce(fn interface{}, defaultValue interface{}, axis int) (retVal *Dense, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}

	if rd, ok := e.(Reducer); ok {
		var val Tensor
		if val, err = rd.Reduce(t, axis, fn, defaultValue); err != nil {
			err = errors.Wrapf(err, opFail, "Dense.Reduce")
			return
		}
		retVal = val.(*Dense)
		return
	}
	return nil, errors.Errorf("Engine %v is not a Reducer", e)
}
