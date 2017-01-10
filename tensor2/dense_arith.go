package tensor

import "github.com/pkg/errors"

func prepDDOp(a, b *Dense, opts ...FuncOpt) (an, bn, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
	var ok bool
	if an, ok = a.data.(Number); !ok {
		err = noopError{}
		return
	}

	if bn, ok = b.data.(Number); !ok {
		err = noopError{}
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}

	safe, incr, reuseT := parseSafeReuse(opts...)
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot add with reuse")
			return
		}

		if rn, ok = reuse.data.(Number); !ok {

		}
	}
	return
}

func addDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepDDOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if reuse == b {
			// b + b first, because b will get clobbered
			if err = rn.Add(bn); err != nil {
				err = errors.Wrapf(err, opFail, "addDD. Unable to add Array `b` to Array `reused`")
				return
			}

			if err = rn.Add(an); err != nil {
				err = errors.Wrapf(err, opFail, "addDD. Unable to add Array `a` to Array `reused`")
				return
			}

			return reuse, nil
		}

		if err = rn.Add(an); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array `a` to Array `reused`")
			return
		}
		if err = rn.Add(bn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array `b` to Array `reused`")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safeAdd(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array `a` and Array `b` to Array `reused`")
			return
		}
		retVal = reuse
	case safe:

	case !safe:
		if err = an.Add(bn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to safely add Array `a` to Array `reused`")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}
