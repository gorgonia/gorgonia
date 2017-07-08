package tensor

import "github.com/pkg/errors"

func prepBinaryDense(a, b DenseTensor, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
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

	fo := ParseFuncOpts(opts...)
	reuseT, incr := fo.IncrReuse()
	safe = fo.Safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != at.Kind() {
			err = errors.Errorf(typeMismatch, at, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != a.Shape().TotalSize() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

// Add performs a + b. The FuncOpts determine what kind of operation it is
func (e StdEng) Add(a, b Tensor, opts ...FuncOpt) (retVal Tensor, err error) {
	// check if the tensors are accessible
	if !a.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, a)
		return
	}

	if !b.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, b)
		return
	}

	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:
			var reuse *Dense
			var safe, toReuse, incr bool
			reuse, safe, toReuse, incr, err = prepBinaryDense(at, bt, opts...)
			if err != nil {
				return nil, err
			}

			if reuse != nil && !reuse.IsNativelyAccessible() {
				err = errors.Errorf(inaccessibleData, reuse)
				return
			}

			// TODO: check masked

			if requiresIterator(at) || requiresIterator(bt) {
				ait := IteratorFromDense(at)
				bit := IteratorFromDense(bt)

				switch {
				case incr:
					iit := NewFlatMaskedIterator(reuse.Info(), nil)
					err = e.E.AddIterIncr(at.rtype(), at.hdr(), bt.hdr(), reuse.hdr(), ait, bit, iit)
				case toReuse:
					copyHeader(reuse.hdr(), at.hdr(), at.rtype())
					err = e.E.AddIter(at.rtype(), reuse.hdr(), bt.hdr(), ait, bit)
					retVal = reuse
				case !safe:
					err = e.E.AddIter(at.rtype(), reuse.hdr(), bt.hdr(), ait, bit)
					retVal = a
				default:
					ret := at.Clone().(DenseTensor)
					err = e.E.AddIter(at.rtype(), ret.hdr(), bt.hdr(), ait, bit)
					retVal = ret
				}
				return
			}

			switch {
			case incr:
				err = e.E.AddIncr(at.rtype(), at.hdr(), bt.hdr(), reuse.hdr())
				retVal = reuse
			case toReuse:
				copyHeader(reuse.hdr(), at.hdr(), at.rtype())
				err = e.E.Add(at.rtype(), reuse.hdr(), bt.hdr())
				retVal = reuse
			case !safe:
				err = e.E.Add(at.rtype(), at.hdr(), bt.hdr())
				retVal = a
			default:
				ret := at.Clone().(DenseTensor)
				err = e.E.Add(at.rtype(), ret.hdr(), bt.hdr())
				retVal = ret
			}
			return

		case *CS:
			return nil, errors.Errorf("NYI")
		default:
			return nil, errors.Errorf("NYI")
		}
	case *CS:
		return nil, errors.Errorf("NYI")
	default:
		return nil, errors.Errorf("NYI")

	}
	panic("Unreachable")
}
