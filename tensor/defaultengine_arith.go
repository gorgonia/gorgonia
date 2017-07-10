package tensor

import "github.com/pkg/errors"

func prepBinaryTensor(a, b Tensor, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	// check if the tensors are accessible
	if !a.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, a)
		return
	}

	if !b.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, b)
		return
	}

	at := a.Dtype()
	bt := b.Dtype()
	if !isNumber(at) && !isNumber(bt) {
		err = noopError{}
		return
	}

	if at.Kind() != bt.Kind() {
		err = errors.Errorf(typeMismatch, at, bt)
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
		return
	}
	return denseFromFuncOpts(a.Shape(), at, opts...)
}

func prepUnaryTensor(a Tensor, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	at := a.Dtype()
	if !isNumber(at) {
		err = noopError{}
		return
	}

	return denseFromFuncOpts(a.Shape(), at, opts...)
}

func denseFromFuncOpts(expShape Shape, expType Dtype, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != expType.Kind() {
			err = errors.Errorf(typeMismatch, expType, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != expShape.TotalSize() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

func prepDataVV(a, b Tensor, reuse *Dense) (dataA, dataB, dataReuse *header, ait, bit, iit Iterator, useIter bool, err error) {
	// prep actual data
	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:
			if requiresIterator(at) || requiresIterator(bt) {
				dataA = at.hdr()
				dataB = bt.hdr()
				ait = IteratorFromDense(at)
				bit = IteratorFromDense(bt)
				if reuse != nil {
					iit = IteratorFromDense(reuse)
					dataReuse = reuse.hdr()
				}

				useIter = true
			} else {
				dataA = at.hdr()
				dataB = bt.hdr()
				if reuse != nil {
					dataReuse = reuse.hdr()
				}
			}
		case *CS:
			dataA = at.hdr()
			dataB = bt.hdr()
			ait = IteratorFromDense(at)
			bit = NewFlatSparseIterator(bt)
			if reuse != nil {
				dataReuse = reuse.hdr()
				iit = IteratorFromDense(reuse)
			}
		default:
			err = errors.Errorf(typeNYI, "prepDataVV", b)
		}
	case *CS:
		switch bt := b.(type) {
		case DenseTensor:
			dataB = at.hdr()
			dataA = bt.hdr()
			bit = NewFlatSparseIterator(at)
			ait = IteratorFromDense(bt)
			if reuse != nil {
				dataReuse = reuse.hdr()
				iit = IteratorFromDense(reuse)
			}
		case *CS:
			err = errors.Errorf(methodNYI, "prepDataVV", "CS-CS")
		default:
			err = errors.Errorf(typeNYI, "prepDataVV", b)
		}
	default:
		err = errors.Errorf(typeNYI, "prepDataVV", a)
	}
	return
}

func prepDataVS(a Tensor, b interface{}, reuse *Dense) (dataA, scalar, dataReuse *header, ait, bit, iit Iterator, useIter bool, err error) {
	scalar = scalarToHeader(b)
	switch at := a.(type) {
	case DenseTensor:
		if requiresIterator(at) {
			ait = IteratorFromDense(at)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			dataA = at.hdr()
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

func prepDataSV(a interface{}, b Tensor, reuse *Dense) (scalar, dataB, dataReuse *header, ait, bit, iit Iterator, useIter bool, err error) {
	scalar = scalarToHeader(a)
	switch bt := b.(type) {
	case DenseTensor:
		if requiresIterator(bt) {
			bit = IteratorFromDense(bt)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			dataB = bt.hdr()
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

// Add performs a + b. The FuncOpts determine what kind of operation it is.
//
//		a :: DenseTensor, a :: DenseTensor -> DenseTensor
//			unsafe overwrites a
//		a :: SparseTensor, b :: DenseTensor -> DenseTensor
//			unsafe overwrites b
//		a :: DenseTensor, b :: SparseTensor -> DenseTensor
//			unsafe overwrites a
//		a :: SparseTEnsor, b :: SparseTensor -> SparseTensor
//			unsafe unsupported
func (e StdEng) Add(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = prepBinaryTensor(a, b, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *header
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
			copyHeader(dataReuse, dataA, typ)
			err = e.E.AddIter(typ, dataReuse, dataB, ait, bit)
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
		copyHeader(dataReuse, dataA, typ)
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

// Trans performs the translation option of a + b
func (e StdEng) Trans(a Tensor, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = prepUnaryTensor(a, opts...); err != nil {
		return
	}

	var ait, iit Iterator
	var dataA, dataReuse *header
	var useIter bool
	scalar := scalarToHeader(b)
	typ := a.Dtype().Type

	switch at := a.(type) {
	case DenseTensor:
		if requiresIterator(at) {
			ait = IteratorFromDense(at)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			useIter = true
		}

	case *CS:
		return nil, errors.Errorf("NYI")
	default:
		return nil, errors.Errorf("NYI")
	}

	if useIter {
		switch {
		case incr:
			err = e.E.AddIterIncr(typ, dataA, scalar, dataReuse, ait, nil, iit)
			retVal = reuse
		case toReuse:
			copyHeader(dataReuse, dataA, typ)
			err = e.E.AddIter(typ, dataReuse, scalar, ait, nil)
			retVal = reuse
		case !safe:
			err = e.E.AddIter(typ, dataA, scalar, ait, nil)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.AddIter(typ, ret.hdr(), scalar, ait, nil)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.AddIncr(typ, dataA, scalar, dataReuse)
		retVal = reuse
	case toReuse:
		copyHeader(dataReuse, dataA, typ)
		err = e.E.Add(typ, dataReuse, scalar)
		retVal = reuse
	case !safe:
		err = e.E.Add(typ, dataA, scalar)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Add(typ, ret.hdr(), scalar)
		retVal = ret.(Tensor)
	}
	return
}

func (e StdEng) Sub(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = prepBinaryTensor(a, b, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.Add")
		return
	}

	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			copyHeader(dataReuse, dataA, typ)
			err = e.E.SubIter(typ, dataReuse, dataB, ait, bit)
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
		copyHeader(dataReuse, dataA, typ)
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

func (e StdEng) TransInv(a Tensor, b interface{}, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = prepUnaryTensor(a, opts...); err != nil {
		return
	}

	typ := a.Dtype().Type
	var ait, iit Iterator
	var dataA, scalar, dataReuse *header
	var useIter bool
	if dataA, scalar, dataReuse, ait, _, iit, useIter, err = prepDataVS(a, b, reuse); err != nil {
		return
	}

	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, dataA, scalar, dataReuse, ait, nil, iit)
			retVal = reuse
		case toReuse:
			copyHeader(dataReuse, dataA, typ)
			err = e.E.SubIter(typ, dataReuse, scalar, ait, nil)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, dataA, scalar, ait, nil)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.SubIter(typ, ret.hdr(), scalar, ait, nil)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, dataA, scalar, dataReuse)
		retVal = reuse
	case toReuse:
		copyHeader(dataReuse, dataA, typ)
		err = e.E.Sub(typ, dataReuse, scalar)
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, dataA, scalar)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.Sub(typ, ret.hdr(), scalar)
		retVal = ret.(Tensor)
	}
	return

}

func (e StdEng) TransInvR(a interface{}, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	var reuse *Dense
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, err = prepUnaryTensor(b, opts...); err != nil {
		return
	}

	typ := b.Dtype().Type
	var bit, iit Iterator
	var scalar, dataB, dataReuse *header
	var useIter bool
	if scalar, dataB, dataReuse, _, bit, iit, useIter, err = prepDataSV(a, b, reuse); err != nil {
		return
	}

	if useIter {
		switch {
		case incr:
			err = e.E.SubIterIncr(typ, scalar, dataB, dataReuse, nil, bit, iit)
			retVal = reuse
		case toReuse:
			copyHeader(dataReuse, dataB, typ)
			err = e.E.SubIter(typ, scalar, dataReuse, nil, bit)
			retVal = reuse
		case !safe:
			err = e.E.SubIter(typ, scalar, dataB, nil, bit)
			retVal = b
		default:
			ret := b.Clone().(headerer)
			err = e.E.SubIter(typ, scalar, ret.hdr(), nil, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.SubIncr(typ, scalar, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		copyHeader(dataReuse, dataB, typ)
		err = e.E.Sub(typ, scalar, dataReuse)
		retVal = reuse
	case !safe:
		err = e.E.Sub(typ, scalar, dataB)
		retVal = b
	default:
		ret := b.Clone().(headerer)
		err = e.E.Sub(typ, scalar, ret.hdr())
		retVal = ret.(Tensor)
	}
	return

}
