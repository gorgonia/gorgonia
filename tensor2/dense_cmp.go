package tensor

import "github.com/pkg/errors"

func prepDenseCmpBinOp(a, b *Dense, opts ...FuncOpt) (ao, bo ElemOrd, reuse *Dense, safe, same, toReuse bool, err error) {
	var ok bool
	if ao, ok = a.data.(ElemOrd); !ok {
		err = noopError{}
		return
	}

	if bo, ok = b.data.(ElemOrd); !ok {
		err = noopError{}
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	if !safe {
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		// coarse type checking. Actual type switching will happen in the op
		if !same {
			var b Boolser
			if b, ok = reuse.data.(Boolser); !ok {
				err = errors.Errorf(typeMismatch, b, reuse.data)
				return
			}
		} else {
			var kal ElemOrd
			if kal, ok = reuse.data.(ElemOrd); !ok {
				err = errors.Errorf(typeMismatch, kal, reuse.data)
				return
			}
		}

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

func prepOneDenseCmp(a *Dense, opts ...FuncOpt) (ao ElemOrd, reuse *Dense, safe, same, toReuse bool, err error) {
	var ok bool
	if ao, ok = a.data.(ElemOrd); !ok {
		err = noopError{}
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	if !safe {
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)

		// coarse type checking. Actual type switching will happen in the op
		if !same {
			var b Boolser
			if b, ok = reuse.data.(Boolser); !ok {
				err = errors.Errorf(typeMismatch, b, reuse.data)
				return
			}
		} else {
			var kal ElemOrd
			if kal, ok = reuse.data.(ElemOrd); !ok {
				err = errors.Errorf(typeMismatch, kal, reuse.data)
				return
			}
		}

		if err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

/* elEq */

func elEqDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.ElEq(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func elEqDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(ao).(ElemOrd)
	if err = tmp.Memset(b); err != nil {
		return
	}

	var arr Array
	if arr, err = ao.ElEq(tmp, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func elEqSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElemOrd)
	if err = tmp.Memset(a); err != nil {
		return
	}

	var arr Array
	if arr, err = tmp.ElEq(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

/* gt */

func gtDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.Gt(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func gtDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(ao).(ElemOrd)
	if err = tmp.Memset(b); err != nil {
		return
	}

	var arr Array
	if arr, err = ao.Gt(tmp, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func gtSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElemOrd)
	if err = tmp.Memset(a); err != nil {
		return
	}

	var arr Array
	if arr, err = tmp.Gt(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

/* gte */

func gteDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.Gte(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func gteDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(ao).(ElemOrd)
	if err = tmp.Memset(b); err != nil {
		return
	}

	var arr Array
	if arr, err = ao.Gte(tmp, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func gteSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElemOrd)
	if err = tmp.Memset(a); err != nil {
		return
	}

	var arr Array
	if arr, err = tmp.Gte(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

/* lt */

func ltDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.Lt(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func ltDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(ao).(ElemOrd)
	if err = tmp.Memset(b); err != nil {
		return
	}

	var arr Array
	if arr, err = ao.Lt(tmp, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func ltSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElemOrd)
	if err = tmp.Memset(a); err != nil {
		return
	}

	var arr Array
	if arr, err = tmp.Lt(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

/* lte */

func lteDD(a, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, bo, reuse, safe, same, toReuse, err := prepDenseCmpBinOp(a, b, opts...)
	if err != nil {
		return nil, err
	}

	var arr Array
	if arr, err = ao.Lte(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func lteDS(a *Dense, b interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	ao, reuse, safe, same, toReuse, err := prepOneDenseCmp(a, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(ao).(ElemOrd)
	if err = tmp.Memset(b); err != nil {
		return
	}

	var arr Array
	if arr, err = ao.Lte(tmp, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(a.t, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, a.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(a.data, arr)
		retVal = a
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}

func lteSD(a interface{}, b *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	bo, reuse, safe, same, toReuse, err := prepOneDenseCmp(b, opts...)
	if err != nil {
		return nil, err
	}

	tmp := cloneArray(bo).(ElemOrd)
	if err = tmp.Memset(a); err != nil {
		return
	}

	var arr Array
	if arr, err = tmp.Lte(bo, same); err != nil {
		return
	}

	switch {
	case toReuse:
		_, err = copyArray(reuse.data, arr)
		retVal = reuse
	case safe && same:
		d := recycledDense(b.t, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case safe && !same:
		d := recycledDense(Bool, b.Shape().Clone())
		_, err = copyArray(d.data, arr)
		retVal = d
	case !safe:
		_, err = copyArray(b.data, arr)
		retVal = b
	default:
		err = errors.Errorf("Impossible state reached: Safe %t, Reuse %t, Same %t", safe, reuse, same)
	}
	return
}
