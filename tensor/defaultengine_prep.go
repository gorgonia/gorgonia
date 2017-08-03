package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

func denseFromFuncOpts(expShape Shape, expType Dtype, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr, same bool, err error) {
	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
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

func prepBinaryTensor(a, b Tensor, typeclass []Dtype, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr, same bool, err error) {
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

	if typeclass != nil && !typeclassCheck(at, typeclass) && !typeclassCheck(bt, typeclass) {
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

func prepUnaryTensor(a Tensor, typeclass []Dtype, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr, same bool, err error) {
	at := a.Dtype()
	if typeclass != nil && !typeclassCheck(at, typeclass) {
		err = noopError{}
		return
	}

	return denseFromFuncOpts(a.Shape(), at, opts...)
}

func prepDataVV(a, b Tensor, reuse *Dense) (dataA, dataB, dataReuse *storage.Header, ait, bit, iit Iterator, useIter bool, err error) {
	// prep actual data
	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:
			if requiresIterator(a.Engine(), at) || requiresIterator(a.Engine(), bt) {
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

func prepDataVS(a Tensor, b interface{}, reuse *Dense) (dataA, dataB, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	dataB = scalarToHeader(b)
	switch at := a.(type) {
	case DenseTensor:
		if requiresIterator(a.Engine(), at) {
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

func prepDataSV(a interface{}, b Tensor, reuse *Dense) (dataA, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
	dataA = scalarToHeader(a)
	switch bt := b.(type) {
	case DenseTensor:
		if requiresIterator(b.Engine(), bt) {
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

func prepDataUnary(a Tensor, reuse *Dense) (dataA, dataReuse *storage.Header, ait, rit Iterator, useIter bool, err error) {
	switch at := a.(type) {
	case DenseTensor:
		dataA = at.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if requiresIterator(a.Engine(), at) {
			ait = IteratorFromDense(at)
			useIter = true
			if reuse != nil {
				rit = IteratorFromDense(reuse)
			}
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}
