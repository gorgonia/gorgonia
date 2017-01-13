package tensor

import "github.com/pkg/errors"

func prepBinaryDense(a, b *Dense, opts ...FuncOpt) (an, bn, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
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

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
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

func prepUnaryDense(a *Dense, opts ...FuncOpt) (an, rn Number, reuse *Dense, safe, toReuse, incr bool, err error) {
	var ok bool
	if an, ok = a.data.(Number); !ok {
		err = noopError{}
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
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
	an, bn, rn, reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if err = an.IncrAdd(bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "addDD. Unable to increment reuse Array")
			return
		}
		retVal = reuse
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
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func addDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = an.IncrTrans(b, rn); err != nil {
			err = errors.Wrapf(err, "addDS. Unable to IncrTrans the Array reuse by b of %T", b)
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
	case !safe:
		err = an.Trans(b)
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func addSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = bn.IncrTrans(a, rn); err != nil {
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
	case !safe:
		err = bn.Trans(a)
		retVal = b
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

/* sub */

func subDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if err = an.IncrSub(bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "subDD. Unable to increment reuse Array")
			return
		}
		retVal = reuse
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
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func subDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = an.IncrTransInv(b, rn); err != nil {
			err = errors.Wrapf(err, "subDS. Unable to IncrTransInv the Array reuse by b of %T", b)
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
	case !safe:
		err = an.TransInv(b)
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func subSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = bn.IncrTransInvR(a, rn); err != nil {
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
	case !safe:
		err = bn.TransInvR(a)
		retVal = b
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

/* mul */

func mulDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if err = an.IncrMul(bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "mulDD. Unable to increment reuse Array")
			return
		}
		retVal = reuse
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
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func mulDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = an.IncrScale(b, rn); err != nil {
			err = errors.Wrapf(err, "mulDS. Unable to IncrScale the Array reuse by b of %T", b)
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
	case !safe:
		err = an.Scale(b)
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func mulSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = bn.IncrScale(a, rn); err != nil {
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
	case !safe:
		err = bn.Scale(a)
		retVal = b
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

/* div */

func divDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if err = an.IncrDiv(bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "divDD. Unable to increment reuse Array")
			return
		}
		retVal = reuse
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
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func divDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = an.IncrScaleInv(b, rn); err != nil {
			err = errors.Wrapf(err, "divDS. Unable to IncrScaleInv the Array reuse by b of %T", b)
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
	case !safe:
		err = an.ScaleInv(b)
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func divSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = bn.IncrScaleInvR(a, rn); err != nil {
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
	case !safe:
		err = bn.ScaleInvR(a)
		retVal = b
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

/* pow */

func powDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	an, bn, rn, reuse, safe, toReuse, incr, err := prepBinaryDense(a, b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		if err = an.IncrPow(bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "powDD. Unable to increment reuse Array")
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safePow(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "powDD. Unable to pow Array a and Array b to Array reused")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safePow(an, bn, rn); err != nil {
			err = errors.Wrapf(err, opFail, "powDD. Unable to safely pow Array a and b to rn")
			return
		}
		return
	case !safe:
		if err = an.Pow(bn); err != nil {
			err = errors.Wrapf(err, opFail, "powDD. Unable to safely pow Array a to Array reused")
			return
		}
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func powDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	an, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(a, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = an.IncrPowOf(b, rn); err != nil {
			err = errors.Wrapf(err, "powDS. Unable to IncrPowOf the Array reuse by b of %T", b)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safePowOf(an, b, rn); err != nil {
			err = errors.Wrapf(err, "powDS")
			return
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(a.t, a.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safePowOf(an, b, rn); err != nil {
			err = errors.Wrapf(err, "powDS. Unable to safely pow ")
			return
		}
	case !safe:
		err = an.PowOf(b)
		retVal = a
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}

func powSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bn, rn, reuse, safe, toReuse, incr, err := prepUnaryDense(b, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		if err = bn.IncrPowOfR(a, rn); err != nil {
			err = errors.Wrapf(err, "powSD. Unable to PowOf the Array reuse by a of %T", a)
			return
		}
		retVal = reuse
	case toReuse:
		if _, err = safePowOfR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "powSD")
		}
		retVal = reuse
	case safe:
		retVal = recycledDense(b.t, b.Shape().Clone())
		rn = retVal.data.(Number)
		if _, err = safePowOfR(bn, a, rn); err != nil {
			err = errors.Wrapf(err, "powSD. Unable to safely pow ")
			return
		}
	case !safe:
		err = bn.PowOfR(a)
		retVal = b
	default:
		err = errors.Errorf("Unknown state reached: Safe %t, Incr %t, Reuse %t", safe, incr, reuse)
	}
	return
}
