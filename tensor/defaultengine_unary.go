package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func (e StdEng) Neg(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, numberTypes); err != nil {
		err = errors.Wrapf(err, "Neg failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
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
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.NegIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Neg")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.NegIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.NegIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.NegIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Neg(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Neg")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Neg(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Neg(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Neg(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Inv(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, numberTypes); err != nil {
		err = errors.Wrapf(err, "Inv failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Inv")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.InvIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Inv")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.InvIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.InvIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.InvIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Inv(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Inv")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Inv(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Inv(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Inv(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Square(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, numberTypes); err != nil {
		err = errors.Wrapf(err, "Square failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Square")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.SquareIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Square")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.SquareIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.SquareIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.SquareIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Square(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Square")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Square(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Square(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Square(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Cube(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, numberTypes); err != nil {
		err = errors.Wrapf(err, "Cube failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Cube")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.CubeIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Cube")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.CubeIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.CubeIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.CubeIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Cube(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Cube")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Cube(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Cube(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Cube(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Exp(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatcmplxTypes); err != nil {
		err = errors.Wrapf(err, "Exp failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Exp")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.ExpIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Exp")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.ExpIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.ExpIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.ExpIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Exp(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Exp")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Exp(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Exp(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Exp(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Tanh(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatcmplxTypes); err != nil {
		err = errors.Wrapf(err, "Tanh failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Tanh")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.TanhIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Tanh")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.TanhIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.TanhIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.TanhIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Tanh(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Tanh")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Tanh(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Tanh(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Tanh(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Log(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatcmplxTypes); err != nil {
		err = errors.Wrapf(err, "Log failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Log")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.LogIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Log")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.LogIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.LogIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.LogIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Log(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Log")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Log(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Log(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Log(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Log2(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatTypes); err != nil {
		err = errors.Wrapf(err, "Log2 failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Log2")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.Log2Iter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Log2")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.Log2Iter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.Log2Iter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.Log2Iter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Log2(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Log2")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Log2(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Log2(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Log2(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Log10(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatcmplxTypes); err != nil {
		err = errors.Wrapf(err, "Log10 failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Log10")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.Log10Iter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Log10")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.Log10Iter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.Log10Iter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.Log10Iter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Log10(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Log10")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Log10(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Log10(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Log10(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Sqrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatcmplxTypes); err != nil {
		err = errors.Wrapf(err, "Sqrt failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Sqrt")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.SqrtIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Sqrt")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.SqrtIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.SqrtIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.SqrtIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Sqrt(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Sqrt")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sqrt(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Sqrt(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Sqrt(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Cbrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatTypes); err != nil {
		err = errors.Wrapf(err, "Cbrt failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Cbrt")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.CbrtIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Cbrt")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.CbrtIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.CbrtIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.CbrtIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Cbrt(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Cbrt")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Cbrt(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Cbrt(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Cbrt(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) InvSqrt(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, floatTypes); err != nil {
		err = errors.Wrapf(err, "InvSqrt failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.InvSqrt")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.InvSqrtIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform InvSqrt")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.InvSqrtIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.InvSqrtIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.InvSqrtIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.InvSqrt(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform InvSqrt")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.InvSqrt(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.InvSqrt(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.InvSqrt(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Abs(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, signedTypes); err != nil {
		err = errors.Wrapf(err, "Abs failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Abs")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.AbsIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Abs")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.AbsIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.AbsIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.AbsIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Abs(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Abs")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Abs(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Abs(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Abs(typ, h.hdr())
		retVal = cloned
	}
	return

}
func (e StdEng) Sign(a Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(a, signedTypes); err != nil {
		err = errors.Wrapf(err, "Sign failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil {
		return nil, errors.Wrapf(err, opFail, "StdEng.Sign")
	}

	if useIter {
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.SignIter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform Sign")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.SignIter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.SignIter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.SignIter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
	switch {
	case incr:
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		if err = e.E.Sign(typ, h.hdr()); err != nil {
			return nil, errors.Wrap(err, "Unable to perform Sign")
		}
		err = e.E.Add(typ, dataReuse, h.hdr())
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sign(typ, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Sign(typ, dataA)
		retVal = a
	default: // safe by default
		cloned := a.Clone().(Tensor)
		h, ok := cloned.(headerer)
		if !ok {
			return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
		}
		err = e.E.Sign(typ, h.hdr())
		retVal = cloned
	}
	return

}
