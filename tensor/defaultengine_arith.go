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

	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	// check if the tensors are accessible
	if !t.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, t)
		return
	}

	if !other.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	switch at := a.(type) {
	case DenseTensor:
		switch bt := b.(type) {
		case DenseTensor:

		case *CS:

		}
	case *CS:

	}
}
