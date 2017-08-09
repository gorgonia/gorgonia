package tensor

import "github.com/pkg/errors"

/*
GENERATED FILE. DO NOT EDIT
*/

func (t *Dense) Gt(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if gter, ok := e.(Gter); ok {
		var ret Tensor
		if ret, err = gter.Gt(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Gt()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Gt - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Gt()")
}

func (t *Dense) Gte(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if gteer, ok := e.(Gteer); ok {
		var ret Tensor
		if ret, err = gteer.Gte(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Gte()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Gte - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Gte()")
}

func (t *Dense) Lt(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if lter, ok := e.(Lter); ok {
		var ret Tensor
		if ret, err = lter.Lt(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Lt()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Lt - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Lt()")
}

func (t *Dense) Lte(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if lteer, ok := e.(Lteer); ok {
		var ret Tensor
		if ret, err = lteer.Lte(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Lte()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Lte - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Lte()")
}

func (t *Dense) ElEq(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if eleqer, ok := e.(ElEqer); ok {
		var ret Tensor
		if ret, err = eleqer.ElEq(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Eq()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Eq - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Eq()")
}

func (t *Dense) ElNe(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if eleqer, ok := e.(ElEqer); ok {
		var ret Tensor
		if ret, err = eleqer.ElNe(t, other, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do Ne()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Ne - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Ne()")
}

func (t *Dense) GtScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if gter, ok := e.(Gter); ok {
		var ret Tensor
		if ret, err = gter.GtScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do GtScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Gt - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Gt()")
}

func (t *Dense) GteScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if gteer, ok := e.(Gteer); ok {
		var ret Tensor
		if ret, err = gteer.GteScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do GteScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Gte - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Gte()")
}

func (t *Dense) LtScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if lter, ok := e.(Lter); ok {
		var ret Tensor
		if ret, err = lter.LtScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do LtScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Lt - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Lt()")
}

func (t *Dense) LteScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if lteer, ok := e.(Lteer); ok {
		var ret Tensor
		if ret, err = lteer.LteScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do LteScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Lte - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Lte()")
}

func (t *Dense) ElEqScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if eleqer, ok := e.(ElEqer); ok {
		var ret Tensor
		if ret, err = eleqer.ElEqScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do EqScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Eq - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Eq()")
}

func (t *Dense) ElNeScalar(other interface{}, leftTensor bool, opts ...FuncOpt) (retVal *Dense, err error) {
	e := t.e
	if e == nil {
		e = StdEng{}
	}

	if eleqer, ok := e.(ElEqer); ok {
		var ret Tensor
		if ret, err = eleqer.ElNeScalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do NeScalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do Ne - Expected a %T. Got %T instead", retVal, ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support Ne()")
}
