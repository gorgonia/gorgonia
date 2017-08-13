package tensor

import "github.com/pkg/errors"

// Apply applies a function to all the values in the tensor.
func (t *Dense) Apply(fn interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}
	if m, ok := e.(Mapper); ok {
		return m.Map(fn, t, opts...)
	}
	return nil, errors.Errorf("Execution engine for %v not a mapper", t)
}

// Reduce applies a reduction function and reduces the values along the given axis.
func (t *Dense) Reduce(fn interface{}, axis int, defaultValue interface{}) (retVal *Dense, err error) {
	var e Engine = t.e
	if e == nil {
		e = StdEng{}
	}

	if rd, ok := e.(Reducer); ok {
		var val Tensor
		if val, err = rd.Reduce(fn, t, axis, defaultValue); err != nil {
			err = errors.Wrapf(err, opFail, "Dense.Reduce")
			return
		}
		retVal = val.(*Dense)
		return
	}
	return nil, errors.Errorf("Engine %v is not a Reducer", e)
}
