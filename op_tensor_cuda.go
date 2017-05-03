// +build cuda

package gorgonia

import (
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
)

func (op repeatOp) CUDADo(extern External, dev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	extern.Signal()
	if retVal, err = op.Do(inputs...); err != nil {
		err = errors.Wrapf(err, "Errored while CUDA Doing")
		return
	}

	if prealloc != nil {
		// check shape
		if !prealloc.Shape().Eq(retVal.Shape()) {
			return nil, errors.Errorf("Shape mismatch. Prealloc has %v and result has %v", prealloc.Shape(), retVal.Shape())
		}

		ret := retVal
		retVal, err = Copy(prealloc, ret)
		returnValue(ret)
		return
	}

	var mem Memory
	var s tensor.Shape
	dt := retVal.Dtype()
	s = retVal.Shape()
	memsize := calcMemSize(dt, s)
	if mem, err = extern.Get(dev, memsize); err != nil {
		return nil, errors.Wrapf(err, allocFail, memsize, dev)
	}

	t := TypeOf(retVal)
	ret := retVal
	if retVal, err = makeValueFromMem(t, s, mem); err != nil {
		return nil, errors.Wrapf(err, "While making value from mem type: %v shape: %s", t, s)
	}
	retVal, err = Copy(retVal, ret)
	returnValue(ret)
	return
}

func (op repeatOp) CallsExtern() bool { return true }
