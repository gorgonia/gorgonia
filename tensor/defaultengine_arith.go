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
	return nil, nil
	// check if the tensors are accessible
	if !a.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !b.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:
			reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
			if err != nil {
				return nil, err
			}

			if reuse != nil && !reuse.IsNativelyAccessible() {
				err = errors.Errorf(inaccessibleData, reuse)
				return
			}

			// TODO: check masked

			if at.IsMaterializable() || bt.IsMaterializable() {
				ait := NewFlatMaskedIterator(at.Info(), nil)
				bit := NewFlatMaskedIterator(bt.Info(), nil)

				switch {
				case incr:
					iit := NewFlatMaskedIterator(reuse.Info(), nil)
					err = e.StdEng.AddIerIncr(at.rtype(), at.hdr(), bt.hdr(), ait, bit, iit)
				case reuse:
					copyHeader(reuse.hdr(), at.hdr())
					err = e.StdEng.AddIter(at.rtype(), reuse.hdr(), bt.hdr(), ait, bit)
					retVal = reuse
				case !safe:
					err = e.StdEng.AddIter(at.rtype(), reuse.hdr(), bt.hdr(), ait, bit)
					retVal = a
				default:
					retVal = at.Clone().(DenseTensor)
					err = e.StdEng.AddIter(at.rtype(), retVal.hdr(), bt.hdr(), ait, bit)
				}
				return
			}

			switch {
			case incr:
				err = e.StdEng.AddIncr(at.rtype(), at.hdr(), bt.hdr(), reuse.hdr())
				retVal = reuse
			case toReuse:
				copyHeader(reuse.hdr(), at.hdr())
				err = e.StdEng.Add(at.rtype(), reuse.hdr(), bt.hdr())
				retVal = reuse
			case !safe:
				err = e.StdEng.Add(at.rtype(), at.hdr(), bt.hdr())
				retVal = a
			default:
				retVal = at.Clone().(DenseTensor)
				err = e.StdEng.Add(at.rtype(), retVal.hdr(), bt.hdr())
			}
			return

		case *CS:
		default:
			return nil, errors.Errorf("NYI")
		}
	case *CS:
	default:
		return nil, errors.Errorf("NYI")

	}
}
