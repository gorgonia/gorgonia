package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

func (e StdEng) Clamp(a Tensor, min, max interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, nonComplexNumberTypes); err != nil {
		return nil, errors.Wrap(err, "Clamp failed")
	}

	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), false, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Neg")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			if err = e.E.ClampIter(typ, cloned.hdr(), ait, min, max); err != nil {
				return nil, errors.Wrapf(err, "Unable to perform Clamp")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, cloned.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.ClampIter(typ, dataReuse, rit, min, max)
			retVal = reuse
		case !safe:
			err = e.E.ClampIter(typ, dataA, ait, min, max)
			retVal = a
		default:
			cloned := a.Clone().(Tensor)
			err = e.E.ClampIter(typ, cloned.hdr(), ait, min, max)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		if err = e.E.Clamp(typ, cloned.hdr(), min, max); err != nil {
			return nil, errors.Wrapf(err, "Unable to perform Clamp")
		}
		err = e.E.Add(typ, dataReuse, cloned.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Clamp(typ, dataReuse, min, max)
		retVal = reuse
	case !safe:
		err = e.E.Clamp(typ, dataA, min, max)
		retVal = a
	default:
		cloned := a.Clone().(Tensor)
		err = e.E.Clamp(typ, cloned.hdr(), min, max)
		retVal = cloned
	}
	return
}
