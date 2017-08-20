package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
)

func handleFuncOpts(expShape Shape, expType Dtype, strict bool, opts ...FuncOpt) (reuse DenseTensor, safe, toReuse, incr, same bool, err error) {
	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	same = fo.Same()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDenseTensor(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if (strict || same) && reuse.Dtype() != expType {
			err = errors.Errorf(typeMismatch, expType, reuse.Dtype())
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

func binaryCheck(a, b Tensor, tc *typeclass) (err error) {
	// check if the tensors are accessible
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}

	if !b.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, b)
	}

	at := a.Dtype()
	bt := b.Dtype()
	if tc != nil {
		if err = typeclassCheck(at, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "a")
		}
		if err = typeclassCheck(bt, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "b")
		}
	}

	if at.Kind() != bt.Kind() {
		return errors.Errorf(typeMismatch, at, bt)
	}
	if !a.Shape().Eq(b.Shape()) {
		return errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
	}
	return nil
}

func unaryCheck(a Tensor, tc *typeclass) error {
	if !a.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, a)
	}
	at := a.Dtype()
	if tc != nil {
		if err := typeclassCheck(at, tc); err != nil {
			return errors.Wrapf(err, typeclassMismatch, "a")
		}
	}
	return nil
}

// prepDataVV prepares the data given the input and reuse tensors. It also retruns several indicators
//
// useIter indicates that the iterator methods should be used.
// swap indicates that the operands are swapped.
func prepDataVV(a, b Tensor, reuse DenseTensor) (dataA, dataB, dataReuse *storage.Header, ait, bit, iit Iterator, useIter, swap bool, err error) {
	// prep actual data
	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:
			if requiresOrderedIterator(a.Engine(), at) || requiresOrderedIterator(a.Engine(), bt) {
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
			useIter = true
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
		useIter = true
		switch bt := b.(type) {
		case DenseTensor:
			swap = true
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

func prepDataVS(a Tensor, b interface{}, reuse DenseTensor) (dataA, dataB, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	dataB = scalarToHeader(b)
	switch at := a.(type) {
	case DenseTensor:
		dataA = at.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if requiresOrderedIterator(a.Engine(), at) {
			ait = IteratorFromDense(at)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

func prepDataSV(a interface{}, b Tensor, reuse DenseTensor) (dataA, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
	dataA = scalarToHeader(a)
	switch bt := b.(type) {
	case DenseTensor:
		dataB = bt.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if requiresOrderedIterator(b.Engine(), bt) {
			bit = IteratorFromDense(bt)
			if reuse != nil {
				iit = IteratorFromDense(reuse)
			}
			useIter = true
		}
	case *CS:
		err = errors.Errorf("NYI")
	default:
		err = errors.Errorf("NYI")
	}
	return
}

func prepDataUnary(a Tensor, reuse DenseTensor) (dataA, dataReuse *storage.Header, ait, rit Iterator, useIter bool, err error) {
	switch at := a.(type) {
	case DenseTensor:
		dataA = at.hdr()
		if reuse != nil {
			dataReuse = reuse.hdr()
		}
		if requiresOrderedIterator(a.Engine(), at) {
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
