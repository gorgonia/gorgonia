package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func (e StdEng) Gt(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Gt failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Gt")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.GtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.GtSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.GtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.GtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.GtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.GtSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.GtSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Gt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.GtSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Gt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) Gte(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Gte failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Gte")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.GteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.GteSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.GteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.GteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.GteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.GteSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.GteSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Gte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.GteSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Gte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) Lt(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Lt failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Lt")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.LtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.LtSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.LtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.LtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.LtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.LtSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.LtSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Lt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.LtSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Lt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) Lte(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Lte failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Lte")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.LteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.LteSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.LteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.LteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.LteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.LteSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.LteSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Lte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.LteSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Lte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) Eq(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Eq failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Eq")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.EqSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.EqSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.EqIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.EqSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.EqIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.EqSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.EqSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Eq(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.EqSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Eq(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) Ne(a Tensor, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = binaryCheck(a, b, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Ne failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(a.Shape(), a.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.Ne")
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.NeSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.NeSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.NeIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.NeSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.NeIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.NeSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.NeSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Ne(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.NeSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Ne(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) GtScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Gt failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Gt")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Gt")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.GtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.GtSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.GtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.GtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.GtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.GtSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.GtSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Gt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.GtSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Gt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) GteScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Gte failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Gte")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Gte")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.GteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.GteSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.GteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.GteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.GteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.GteSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.GteSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Gte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.GteSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Gte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) LtScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Lt failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Lt")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Lt")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.LtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.LtSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.LtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.LtSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.LtIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.LtSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.LtSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Lt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.LtSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Lt(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) LteScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Lte failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Lte")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Lte")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.LteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.LteSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.LteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.LteSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.LteIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.LteSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.LteSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Lte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.LteSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Lte(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) EqScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Eq failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Eq")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Eq")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.EqSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.EqSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.EqIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.EqSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.EqIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.EqSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.EqSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Eq(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.EqSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Eq(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}

func (e StdEng) NeScalar(t Tensor, s interface{}, leftTensor bool, opts ...FuncOpt) (retVal Tensor, err error) {
	if err = unaryCheck(t, ordTypes); err != nil {
		return nil, errors.Wrapf(err, "Ne failed")
	}

	var reuse *Dense
	var safe, toReuse, same bool
	if reuse, safe, toReuse, _, same, err = handleFuncOpts(t.Shape(), t.Dtype(), opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit, iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Ne")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.Ne")
		}
	}

	if !same && !toReuse {
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !toReuse && same:
			err = e.E.NeSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case toReuse && same:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.NeSameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !same:
			err = e.E.NeIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case !safe:
			err = e.E.NeSameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			err = e.E.NeIter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
	case !toReuse && same:
		err = e.E.NeSame(typ, dataA, dataB)
		retVal = a
	case toReuse && same:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.NeSame(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !same:
		err = e.E.Ne(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.NeSame(typ, dataA, dataB)
		retVal = a
	default:
		err = e.E.Ne(typ, dataA, dataB, dataReuse)
		retVal = reuse
	}
	return
}
