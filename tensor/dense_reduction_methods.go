package tensor

import "github.com/pkg/errors"

func (t *Dense) Sum(along ...int) (retVal *Dense, err error) {
	var e Engine = t.e
	if sumer, ok := e.(Sumer); ok {
		var ret Tensor
		if ret, err = sumer.Sum(t, along...); err != nil {
			return
		}
		return ret.(*Dense), nil
	}
	return nil, errors.Errorf("Engine does not support Sum")
}

func (t *Dense) Max(along ...int) (retVal *Dense, err error) {
	var e Engine = t.e
	if maxer, ok := e.(Maxer); ok {
		var ret Tensor
		if ret, err = maxer.Max(t, along...); err != nil {
			return
		}
		return ret.(*Dense), nil
	}
	return nil, errors.Errorf("Engine does not support Max")
}

func (t *Dense) Min(along ...int) (retVal *Dense, err error) {
	var e Engine = t.e
	if miner, ok := e.(Miner); ok {
		var ret Tensor
		if ret, err = miner.Min(t, along...); err != nil {
			return
		}
		return ret.(*Dense), nil
	}
	return nil, errors.Errorf("Engine does not support Min")
}
