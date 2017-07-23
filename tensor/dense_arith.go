package tensor

import "github.com/pkg/errors"

/*
GENERATED FILE. DO NOT EDIT
*/

func (t *Dense) Add(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if adder, ok := e.(Adder); ok {
		var ret Tensor
		if ret, err = adder.Add(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Add()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Add()")
}

func (t *Dense) Sub(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if suber, ok := e.(Suber); ok {
		var ret Tensor
		if ret, err = suber.Sub(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Sub()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Sub()")
}

func (t *Dense) Mul(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if muler, ok := e.(Muler); ok {
		var ret Tensor
		if ret, err = muler.Mul(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Mul()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Mul()")
}

func (t *Dense) Div(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if diver, ok := e.(Diver); ok {
		var ret Tensor
		if ret, err = diver.Div(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Div()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Div()")
}

func (t *Dense) Pow(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if power, ok := e.(Power); ok {
		var ret Tensor
		if ret, err = power.Pow(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Pow()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Pow()")
}

func (t *Dense) Mod(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if moder, ok := e.(Moder); ok {
		var ret Tensor
		if ret, err = moder.Mod(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Mod()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Mod()")
}

func (t *Dense) AddScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if adder, ok := e.(Adder); ok {
		var ret Tensor
		if ret, err = adder.AddScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do AddScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Add()")
}

func (t *Dense) SubScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if suber, ok := e.(Suber); ok {
		var ret Tensor
		if ret, err = suber.SubScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do SubScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Sub()")
}

func (t *Dense) MulScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if muler, ok := e.(Muler); ok {
		var ret Tensor
		if ret, err = muler.MulScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do MulScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Mul()")
}

func (t *Dense) DivScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if diver, ok := e.(Diver); ok {
		var ret Tensor
		if ret, err = diver.DivScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do DivScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Div()")
}

func (t *Dense) PowScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if power, ok := e.(Power); ok {
		var ret Tensor
		if ret, err = power.PowScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do PowScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Pow()")
}

func (t *Dense) ModScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if moder, ok := e.(Moder); ok {
		var ret Tensor
		if ret, err = moder.ModScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do ModScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Mod()")
}
