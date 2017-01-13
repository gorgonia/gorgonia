package tensor

import (
	"github.com/chewxy/vecf32"
	"github.com/chewxy/vecf64"
	"github.com/pkg/errors"
)

func cloneArray(a Array) Array {
	switch at := a.(type) {
	case f64s:
		retVal := make(f64s, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case f32s:
		retVal := make(f32s, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case ints:
		retVal := make(ints, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case i64s:
		retVal := make(i64s, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case i32s:
		retVal := make(i32s, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case u8s:
		retVal := make(u8s, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	case bs:
		retVal := make(bs, a.Len(), a.Cap())
		copy(retVal, at)
		return retVal
	}
	panic("Unreachable")
}

func prepBinaryArray(a, b Number, optional ...Number) (retVal Number, err error) {
	var reuse Number
	if a.Len() != b.Len() {
		return nil, errors.Errorf(lenMismatch, a.Len(), b.Len())
	}

	if len(optional) >= 1 {
		reuse = optional[0]
		if reuse.Len() != b.Len() {
			return nil, errors.Errorf(lenMismatch, a.Len(), reuse.Len())
		}
	}

	if reuse != nil {
		retVal = reuse
		_, err = copyArray(reuse, a)
	} else {
		retVal = cloneArray(a).(Number)
	}
	return
}

func safeAdd(a, b Number, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepBinaryArray(a, b, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeAdd")
	}

	if err = retVal.Add(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeAdd")
	}
	return
}

func safeSub(a, b Number, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepBinaryArray(a, b, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeSub")
	}

	if err = retVal.Sub(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeSub")
	}
	return
}

func safeMul(a, b Number, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepBinaryArray(a, b, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeDiv")
	}

	if err = retVal.Mul(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeDiv")
	}
	return
}

func safeDiv(a, b Number, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepBinaryArray(a, b, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeDiv")
	}

	if err = retVal.Div(b); err != nil {
		// if it's a math error, we'll pass it back
		if _, ok := err.(MathError); ok {
			return retVal, err
		}
		return nil, errors.Wrapf(err, opFail, "safeDiv")
	}
	return
}

func safePow(a, b Number, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepBinaryArray(a, b, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePow")
	}
	if err = retVal.Pow(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePow")
	}
	return
}

func prepUnaryArray(a Number, optional ...Number) (retVal Number, err error) {
	var reuse Number
	if len(optional) >= 1 {
		reuse = optional[0]
		if reuse.Len() != a.Len() {
			return nil, errors.Errorf(lenMismatch, a.Len(), reuse.Len())
		}
	}

	if reuse != nil {
		_, err = copyArray(reuse, a)
		retVal = reuse
	} else {
		retVal = cloneArray(a).(Number)
	}
	return
}

func safeTrans(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTrans")
	}

	if err = retVal.Trans(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTrans")
	}
	return
}

func safeTransInv(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTransInv")
	}

	if err = retVal.TransInv(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTransInv")
	}
	return
}

func safeTransInvR(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTransInvR")
	}

	if err = retVal.TransInvR(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeTransInvR")
	}
	return
}

func safeScale(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScale")
	}

	if err = retVal.Scale(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScale")
	}
	return
}

func safeScaleInv(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScaleInv")
	}

	if err = retVal.ScaleInv(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScaleInv")
	}
	return
}

func safeScaleInvR(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScaleInvR")
	}

	if err = retVal.ScaleInvR(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeScaleInvR")
	}
	return
}

func safePowOf(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePowOf")
	}

	if err = retVal.PowOf(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePowOf")
	}
	return
}

func safePowOfR(a Number, b interface{}, optional ...Number) (retVal Number, err error) {
	if retVal, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePowOfR")
	}

	if err = retVal.PowOfR(b); err != nil {
		return nil, errors.Wrapf(err, opFail, "safePowOfR")
	}
	return
}

/* Unaries */

// safeSqrt only works on Float s for now
func safeSqrt(a Number, optional ...Number) (retVal Float, err error) {
	var retN Number
	if retN, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeSqrt")
	}

	var ok bool
	if retVal, ok = retN.(Float); !ok {
		return nil, errors.Errorf("Sqrt only works on Floats")
	}

	switch rt := retVal.(type) {
	case Float64ser:
		vecf64.Sqrt(rt.Float64s())
	case Float32ser:
		vecf32.Sqrt(rt.Float32s())
	}
	return
}

func safeInvSqrt(a Number, optional ...Number) (retVal Number, err error) {
	var retN Number
	if retN, err = prepUnaryArray(a, optional...); err != nil {
		return nil, errors.Wrapf(err, opFail, "safeSqrt")
	}

	var ok bool
	if retVal, ok = retN.(Float); !ok {
		return nil, errors.Errorf("Sqrt only works on Floats")
	}

	switch rt := retVal.(type) {
	case Float64ser:
		vecf64.Sqrt(rt.Float64s())
	case Float32ser:
		vecf32.Sqrt(rt.Float32s())
	}
	return
}
