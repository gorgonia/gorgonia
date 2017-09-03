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
			returnOpOpt(fo)
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse != nil && !reuse.IsNativelyAccessible() {
			returnOpOpt(fo)
			err = errors.Errorf(inaccessibleData, reuse)
			return
		}

		if (strict || same) && reuse.Dtype() != expType {
			returnOpOpt(fo)
			err = errors.Errorf(typeMismatch, expType, reuse.Dtype())
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != expShape.TotalSize() && !expShape.IsScalar() {
			returnOpOpt(fo)
			err = errors.Errorf(shapeMismatch, reuse.Shape(), expShape)
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch - reuse.len() %v, expShape.TotalSize() %v", reuse.len(), expShape.TotalSize())
			return
		}

	}
	returnOpOpt(fo)
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
func prepDataVV(a, b Tensor, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, ait, bit, iit Iterator, useIter, swap bool, err error) {
	// get data
	// var ah, bh headerer
	// var ok bool
	// if ah, ok = a.(headerer); !ok {
	// 	err = errors.New("Unable to get *storage.Header from a")
	// 	return
	// }
	// if bh, ok = b.(headerer); !ok {
	// 	err = errors.New("Unable to get *storage.Header from b")
	// 	return
	// }
	dataA = a.hdr()
	dataB = b.hdr()
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// iter
	useIter = a.RequiresIterator() || b.RequiresIterator() || (reuse != nil && reuse.RequiresIterator())
	if useIter {
		ait = a.Iterator()
		bit = b.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
	}

	// swap
	if _, ok := a.(*CS); ok {
		if _, ok := b.(DenseTensor); ok {
			swap = true
			dataA, dataB = dataB, dataA
			ait, bit = bit, ait
		}
	}

	return
}

func prepDataVS(a Tensor, b interface{}, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, ait, iit Iterator, useIter bool, err error) {
	// get data
	// if ah, ok := a.(headerer); ok {
	// 	dataA = ah.hdr()
	// } else {
	// 	err = errors.New("Unable to get data from a")
	// 	return
	// }
	dataA = a.hdr()
	dataB = scalarToHeader(b)
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	if a.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		ait = a.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
		useIter = true
	}
	return
}

func prepDataSV(a interface{}, b Tensor, reuse Tensor) (dataA, dataB, dataReuse *storage.Header, bit, iit Iterator, useIter bool, err error) {
	// get data
	dataA = scalarToHeader(a)
	dataB = b.hdr()
	// if bh, ok := b.(headerer); ok {
	// 	dataB = bh.hdr()
	// } else {
	// 	err = errors.New("Unable to get data from b")
	// 	return
	// }
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// get iterator
	if b.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		bit = b.Iterator()
		if reuse != nil {
			iit = reuse.Iterator()
		}
		useIter = true
	}
	return
}

func prepDataUnary(a Tensor, reuse Tensor) (dataA, dataReuse *storage.Header, ait, rit Iterator, useIter bool, err error) {
	// get data
	if ah, ok := a.(headerer); ok {
		dataA = ah.hdr()
	} else {
		err = errors.New("Unable to get data from a")
		return
	}
	if reuse != nil {
		dataReuse = reuse.hdr()
	}

	// get iterator
	if a.RequiresIterator() || (reuse != nil && reuse.RequiresIterator()) {
		ait = a.Iterator()
		if reuse != nil {
			rit = reuse.Iterator()
		}
		useIter = true
	}
	return
}
