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
			err = errors.Errorf("Reuse is not a number")
			return
		}
	}
	return
}

func prepSD(a *Dense, opts ...FuncOpt) (an, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
	var ok bool
	if an, ok = a.data.(Number); !ok {
		err = noopError{}
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
			err = errors.Errorf("Reuse is not a number")
			return
		}
	}
	return

}



/* add */

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
				err = errors.Wrapf(err, opFail, "addDD. Unable to add Array b to Array reused")
				return
			}

			if err = rn.Add(an); err != nil {
				err = errors.Wrapf(err, opFail, "addDD. Unable to add Array a to Array reused")
				return
			}

			return reuse, nil
		}

		if err = rn.Add(an); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array a to Array reused")
			return
		}
		if err = rn.Add(bn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array b to Array reused")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safeAdd(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to add Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeAdd(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to safely add Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.Add(bn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to safely add Array a to Array reused")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func addDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepSD(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Add(an); err != nil {
			err = errors.Wrapf(err, "addDS. Unable to add Array a to the reuse")
			return
		}
		if err = rn.Trans(b); err != nil {
			err = errors.Wrapf(err, "addDS. Unable to Trans the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeTrans(an, b, rn); err != nil {
			err = errors.Wrapf(err, "addDS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeTrans(an, b, rn); err != nil {
			err = errors.Wrapf(err, "addDS. Unable to safely add ")
			return
		}
		return
	case !safe:
		err = an.Trans(b)
		retVal = a
		return
	}
	panic("Unreachable")
}

func addSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepSD(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Add(bn); err != nil {
			err = errors.Wrapf(err, "addSD. Unable to add Array b to the reuse")
			return
		}
		if err = rn.Trans(a); err != nil {
			err = errors.Wrapf(err, "addSD. Unable to Trans the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeTrans(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "addSD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeTrans(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "addSD. Unable to safely add ")
			return
		}
		return
	case !safe:
		err = bn.Trans(a)
		retVal = b
		return
	}
	panic("Unreachable")
}



/* sub */

func subDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepDDOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if reuse == b {
			// b + b first, because b will get clobbered
			if err = rn.Sub(bn); err != nil {
				err = errors.Wrapf(err, opFail, "subDD. Unable to sub Array b to Array reused")
				return
			}

			if err = rn.Sub(an); err != nil {
				err = errors.Wrapf(err, opFail, "subDD. Unable to sub Array a to Array reused")
				return
			}

			return reuse, nil
		}

		if err = rn.Sub(an); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to sub Array a to Array reused")
			return
		}
		if err = rn.Sub(bn); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to sub Array b to Array reused")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safeSub(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to sub Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeSub(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to safely sub Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.Sub(bn); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to safely sub Array a to Array reused")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func subDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepSD(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Sub(an); err != nil {
			err = errors.Wrapf(err, "subDS. Unable to sub Array a to the reuse")
			return
		}
		if err = rn.TransInv(b); err != nil {
			err = errors.Wrapf(err, "subDS. Unable to TransInv the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeTransInv(an, b, rn); err != nil {
			err = errors.Wrapf(err, "subDS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeTransInv(an, b, rn); err != nil {
			err = errors.Wrapf(err, "subDS. Unable to safely sub ")
			return
		}
		return
	case !safe:
		err = an.TransInv(b)
		retVal = a
		return
	}
	panic("Unreachable")
}

func subSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepSD(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Sub(bn); err != nil {
			err = errors.Wrapf(err, "subSD. Unable to sub Array b to the reuse")
			return
		}
		if err = rn.TransInvR(a); err != nil {
			err = errors.Wrapf(err, "subSD. Unable to TransInv the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeTransInvR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "subSD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeTransInvR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "subSD. Unable to safely sub ")
			return
		}
		return
	case !safe:
		err = bn.TransInvR(a)
		retVal = b
		return
	}
	panic("Unreachable")
}



/* mul */

func mulDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepDDOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if reuse == b {
			// b + b first, because b will get clobbered
			if err = rn.Mul(bn); err != nil {
				err = errors.Wrapf(err, opFail, "mulDD. Unable to mul Array b to Array reused")
				return
			}

			if err = rn.Mul(an); err != nil {
				err = errors.Wrapf(err, opFail, "mulDD. Unable to mul Array a to Array reused")
				return
			}

			return reuse, nil
		}

		if err = rn.Mul(an); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to mul Array a to Array reused")
			return
		}
		if err = rn.Mul(bn); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to mul Array b to Array reused")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safeMul(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to mul Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeMul(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to safely mul Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.Mul(bn); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to safely mul Array a to Array reused")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func mulDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepSD(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Mul(an); err != nil {
			err = errors.Wrapf(err, "mulDS. Unable to mul Array a to the reuse")
			return
		}
		if err = rn.Scale(b); err != nil {
			err = errors.Wrapf(err, "mulDS. Unable to Scale the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeScale(an, b, rn); err != nil {
			err = errors.Wrapf(err, "mulDS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeScale(an, b, rn); err != nil {
			err = errors.Wrapf(err, "mulDS. Unable to safely mul ")
			return
		}
		return
	case !safe:
		err = an.Scale(b)
		retVal = a
		return
	}
	panic("Unreachable")
}

func mulSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepSD(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Mul(bn); err != nil {
			err = errors.Wrapf(err, "mulSD. Unable to mul Array b to the reuse")
			return
		}
		if err = rn.Scale(a); err != nil {
			err = errors.Wrapf(err, "mulSD. Unable to Scale the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeScale(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "mulSD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeScale(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "mulSD. Unable to safely mul ")
			return
		}
		return
	case !safe:
		err = bn.Scale(a)
		retVal = b
		return
	}
	panic("Unreachable")
}



/* div */

func divDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepDDOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if reuse == b {
			// b + b first, because b will get clobbered
			if err = rn.Div(bn); err != nil {
				err = errors.Wrapf(err, opFail, "divDD. Unable to div Array b to Array reused")
				return
			}

			if err = rn.Div(an); err != nil {
				err = errors.Wrapf(err, opFail, "divDD. Unable to div Array a to Array reused")
				return
			}

			return reuse, nil
		}

		if err = rn.Div(an); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to div Array a to Array reused")
			return
		}
		if err = rn.Div(bn); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to div Array b to Array reused")
			return
		}

		return reuse, nil
	case toReuse:
		if _, err = safeDiv(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to div Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeDiv(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to safely div Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.Div(bn); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to safely div Array a to Array reused")
			return
		}
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func divDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepSD(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Div(an); err != nil {
			err = errors.Wrapf(err, "divDS. Unable to div Array a to the reuse")
			return
		}
		if err = rn.ScaleInv(b); err != nil {
			err = errors.Wrapf(err, "divDS. Unable to ScaleInv the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeScaleInv(an, b, rn); err != nil {
			err = errors.Wrapf(err, "divDS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeScaleInv(an, b, rn); err != nil {
			err = errors.Wrapf(err, "divDS. Unable to safely div ")
			return
		}
		return
	case !safe:
		err = an.ScaleInv(b)
		retVal = a
		return
	}
	panic("Unreachable")
}

func divSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepSD(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = rn.Div(bn); err != nil {
			err = errors.Wrapf(err, "divSD. Unable to div Array b to the reuse")
			return
		}
		if err = rn.ScaleInvR(a); err != nil {
			err = errors.Wrapf(err, "divSD. Unable to ScaleInv the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safeScaleInvR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "divSD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safeScaleInvR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "divSD. Unable to safely div ")
			return
		}
		return
	case !safe:
		err = bn.ScaleInvR(a)
		retVal = b
		return
	}
	panic("Unreachable")
}



