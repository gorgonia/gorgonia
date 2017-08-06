package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func (e StdEng) Add(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Add")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.AddIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.AddIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.AddIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.AddIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.AddIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Add(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Add(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Add(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Sub(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Sub")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.SubIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.SubIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sub(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Sub(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Mul(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Mul")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.MulIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.MulIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.MulIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.MulIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.MulIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mul(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Mul(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Mul(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Div(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Div")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.DivIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.DivIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.DivIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.DivIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.DivIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Div(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Div(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Div(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Pow(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Pow")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.PowIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.PowIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.PowIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.PowIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.PowIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Pow(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Pow(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Pow(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Mod(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepBinaryTensor(a, b, numberTypes, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Mod")
		return
	}
	if useIter {
		switch {
		case incr:
			err = e.E.ModIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.ModIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case !safe:
			err = e.E.ModIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.ModIter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.ModIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mod(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.Mod(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Mod(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) AddScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Add")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Add")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.AddIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.AddIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.AddIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.AddIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.AddIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.AddIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.AddIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Add(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Add(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Add(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Add(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Add(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) SubScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Sub")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Sub")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.SubIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.SubIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.SubIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.SubIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Sub(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Sub(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Sub(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Sub(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) MulScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Mul")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Mul")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.MulIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.MulIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.MulIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.MulIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.MulIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.MulIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.MulIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mul(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Mul(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Mul(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Mul(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Mul(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) DivScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Div")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Div")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.DivIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.DivIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.DivIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.DivIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.DivIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.DivIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.DivIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Div(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Div(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Div(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Div(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Div(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) PowScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Pow")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Pow")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.PowIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.PowIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.PowIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.PowIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.PowIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.PowIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.PowIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Pow(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Pow(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Pow(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Pow(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Pow(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) ModScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = prepUnaryTensor(t, numberTypes, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Mod")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, opFail, "StdEng.Mod")
			return
		}
	}

	if useIter {
		switch {
		case incr:
			err = e.E.ModIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.ModIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			ait.Reset()
			iit.Reset()
			err = e.E.ModIter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		case !safe:
			err = e.E.ModIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.ModIter(typ, ret.hdr(), dataB, ait, bit)
			} else {
				err = e.E.ModIter(typ, dataA, ret.hdr(), ait, bit)
			}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.ModIncr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.Mod(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.Mod(typ, dataA, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Mod(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		if leftTensor {
			err = e.E.Mod(typ, ret.hdr(), dataB)
		} else {
			err = e.E.Mod(typ, dataA, ret.hdr())
		}
		retVal = ret.(Tensor)
	}
	return
}
