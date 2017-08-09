package tensor

import "github.com/pkg/errors"

func (e StdEng) Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	if !a.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, a)
	}
	h, ok := a.(headerer)
	if !ok {
		return nil, errors.Errorf(inaccessibleData, a)
	}

	if requiresIterator(a) {
		ait := IteratorFromTensor(a)
		err = e.E.ClampIter(a.Dtype().Type, h.hdr(), ait, min, max)
		retVal = a
		return
	}
	err = e.E.Clamp(a.Dtype().Type, h.hdr(), min, max)
	retVal = a
	return
}
