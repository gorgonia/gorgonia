package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func prepBinaryDense(a, b *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) && !isNumber(b.t) {
		err = noopError{}
		return
	}
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(typeMismatch, a.t, b.t)
		return
	}
	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind() {
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != a.len() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

func prepUnaryDense(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) {
		err = noopError{}
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind() {
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return
		}

		if reuse.len() != a.len() {
			err = errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

/* Add */

func (t *Dense) Add(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
	case toReuse:
		copyDense(reuse, t)
		reuse.add(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.add(other)
	case !safe:
		t.add(other)
		retVal = t
	}
	return
}
func (t *Dense) add(other *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		addI(t.ints(), other.ints())
	case reflect.Int8:
		addI8(t.int8s(), other.int8s())
	case reflect.Int16:
		addI16(t.int16s(), other.int16s())
	case reflect.Int32:
		addI32(t.int32s(), other.int32s())
	case reflect.Int64:
		addI64(t.int64s(), other.int64s())
	case reflect.Uint:
		addU(t.uints(), other.uints())
	case reflect.Uint8:
		addU8(t.uint8s(), other.uint8s())
	case reflect.Uint16:
		addU16(t.uint16s(), other.uint16s())
	case reflect.Uint32:
		addU32(t.uint32s(), other.uint32s())
	case reflect.Uint64:
		addU64(t.uint64s(), other.uint64s())
	case reflect.Float32:
		addF32(t.float32s(), other.float32s())
	case reflect.Float64:
		addF64(t.float64s(), other.float64s())
	case reflect.Complex64:
		addC64(t.complex64s(), other.complex64s())
	case reflect.Complex128:
		addC128(t.complex128s(), other.complex128s())
	}
}

/* Sub */

func (t *Dense) Sub(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
	case toReuse:
		copyDense(reuse, t)
		reuse.sub(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.sub(other)
	case !safe:
		t.sub(other)
		retVal = t
	}
	return
}
func (t *Dense) sub(other *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		subI(t.ints(), other.ints())
	case reflect.Int8:
		subI8(t.int8s(), other.int8s())
	case reflect.Int16:
		subI16(t.int16s(), other.int16s())
	case reflect.Int32:
		subI32(t.int32s(), other.int32s())
	case reflect.Int64:
		subI64(t.int64s(), other.int64s())
	case reflect.Uint:
		subU(t.uints(), other.uints())
	case reflect.Uint8:
		subU8(t.uint8s(), other.uint8s())
	case reflect.Uint16:
		subU16(t.uint16s(), other.uint16s())
	case reflect.Uint32:
		subU32(t.uint32s(), other.uint32s())
	case reflect.Uint64:
		subU64(t.uint64s(), other.uint64s())
	case reflect.Float32:
		subF32(t.float32s(), other.float32s())
	case reflect.Float64:
		subF64(t.float64s(), other.float64s())
	case reflect.Complex64:
		subC64(t.complex64s(), other.complex64s())
	case reflect.Complex128:
		subC128(t.complex128s(), other.complex128s())
	}
}

/* Mul */

func (t *Dense) Mul(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
	case toReuse:
		copyDense(reuse, t)
		reuse.mul(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.mul(other)
	case !safe:
		t.mul(other)
		retVal = t
	}
	return
}
func (t *Dense) mul(other *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		mulI(t.ints(), other.ints())
	case reflect.Int8:
		mulI8(t.int8s(), other.int8s())
	case reflect.Int16:
		mulI16(t.int16s(), other.int16s())
	case reflect.Int32:
		mulI32(t.int32s(), other.int32s())
	case reflect.Int64:
		mulI64(t.int64s(), other.int64s())
	case reflect.Uint:
		mulU(t.uints(), other.uints())
	case reflect.Uint8:
		mulU8(t.uint8s(), other.uint8s())
	case reflect.Uint16:
		mulU16(t.uint16s(), other.uint16s())
	case reflect.Uint32:
		mulU32(t.uint32s(), other.uint32s())
	case reflect.Uint64:
		mulU64(t.uint64s(), other.uint64s())
	case reflect.Float32:
		mulF32(t.float32s(), other.float32s())
	case reflect.Float64:
		mulF64(t.float64s(), other.float64s())
	case reflect.Complex64:
		mulC64(t.complex64s(), other.complex64s())
	case reflect.Complex128:
		mulC128(t.complex128s(), other.complex128s())
	}
}

/* Div */

func (t *Dense) Div(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
	case toReuse:
		copyDense(reuse, t)
		reuse.div(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.div(other)
	case !safe:
		t.div(other)
		retVal = t
	}
	return
}
func (t *Dense) div(other *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		divI(t.ints(), other.ints())
	case reflect.Int8:
		divI8(t.int8s(), other.int8s())
	case reflect.Int16:
		divI16(t.int16s(), other.int16s())
	case reflect.Int32:
		divI32(t.int32s(), other.int32s())
	case reflect.Int64:
		divI64(t.int64s(), other.int64s())
	case reflect.Uint:
		divU(t.uints(), other.uints())
	case reflect.Uint8:
		divU8(t.uint8s(), other.uint8s())
	case reflect.Uint16:
		divU16(t.uint16s(), other.uint16s())
	case reflect.Uint32:
		divU32(t.uint32s(), other.uint32s())
	case reflect.Uint64:
		divU64(t.uint64s(), other.uint64s())
	case reflect.Float32:
		divF32(t.float32s(), other.float32s())
	case reflect.Float64:
		divF64(t.float64s(), other.float64s())
	case reflect.Complex64:
		divC64(t.complex64s(), other.complex64s())
	case reflect.Complex128:
		divC128(t.complex128s(), other.complex128s())
	}
}

/* Pow */

func (t *Dense) Pow(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
	case toReuse:
		copyDense(reuse, t)
		reuse.pow(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.pow(other)
	case !safe:
		t.pow(other)
		retVal = t
	}
	return
}
func (t *Dense) pow(other *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		powI(t.ints(), other.ints())
	case reflect.Int8:
		powI8(t.int8s(), other.int8s())
	case reflect.Int16:
		powI16(t.int16s(), other.int16s())
	case reflect.Int32:
		powI32(t.int32s(), other.int32s())
	case reflect.Int64:
		powI64(t.int64s(), other.int64s())
	case reflect.Uint:
		powU(t.uints(), other.uints())
	case reflect.Uint8:
		powU8(t.uint8s(), other.uint8s())
	case reflect.Uint16:
		powU16(t.uint16s(), other.uint16s())
	case reflect.Uint32:
		powU32(t.uint32s(), other.uint32s())
	case reflect.Uint64:
		powU64(t.uint64s(), other.uint64s())
	case reflect.Float32:
		powF32(t.float32s(), other.float32s())
	case reflect.Float64:
		powF64(t.float64s(), other.float64s())
	case reflect.Complex64:
		powC64(t.complex64s(), other.complex64s())
	case reflect.Complex128:
		powC128(t.complex128s(), other.complex128s())
	}
}

/* Trans */

func (t *Dense) Trans(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.trans(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.trans(other)
	case !safe:
		t.trans(other)
		retVal = t
	}
	return
}
func (t *Dense) trans(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		transI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		transI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		transI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		transI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		transI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		transU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		transU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		transU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		transU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		transU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		transF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		transF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		transC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		transC128(t.complex128s(), b)
	}
}

/* TransInv */

func (t *Dense) TransInv(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.transinv(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.transinv(other)
	case !safe:
		t.transinv(other)
		retVal = t
	}
	return
}
func (t *Dense) transinv(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		transinvI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		transinvI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		transinvI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		transinvI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		transinvI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		transinvU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		transinvU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		transinvU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		transinvU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		transinvU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		transinvF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		transinvF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		transinvC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		transinvC128(t.complex128s(), b)
	}
}

/* TransInvR */

func (t *Dense) TransInvR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.transinvr(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.transinvr(other)
	case !safe:
		t.transinvr(other)
		retVal = t
	}
	return
}
func (t *Dense) transinvr(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		transinvrI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		transinvrI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		transinvrI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		transinvrI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		transinvrI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		transinvrU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		transinvrU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		transinvrU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		transinvrU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		transinvrU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		transinvrF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		transinvrF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		transinvrC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		transinvrC128(t.complex128s(), b)
	}
}

/* Scale */

func (t *Dense) Scale(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.scale(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.scale(other)
	case !safe:
		t.scale(other)
		retVal = t
	}
	return
}
func (t *Dense) scale(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		scaleI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		scaleI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		scaleI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		scaleI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		scaleI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		scaleU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		scaleU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		scaleU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		scaleU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		scaleU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		scaleF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		scaleF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		scaleC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		scaleC128(t.complex128s(), b)
	}
}

/* ScaleInv */

func (t *Dense) ScaleInv(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.scaleinv(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.scaleinv(other)
	case !safe:
		t.scaleinv(other)
		retVal = t
	}
	return
}
func (t *Dense) scaleinv(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		scaleinvI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		scaleinvI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		scaleinvI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		scaleinvI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		scaleinvI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		scaleinvU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		scaleinvU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		scaleinvU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		scaleinvU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		scaleinvU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		scaleinvF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		scaleinvF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		scaleinvC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		scaleinvC128(t.complex128s(), b)
	}
}

/* ScaleInvR */

func (t *Dense) ScaleInvR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.scaleinvr(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.scaleinvr(other)
	case !safe:
		t.scaleinvr(other)
		retVal = t
	}
	return
}
func (t *Dense) scaleinvr(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		scaleinvrI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		scaleinvrI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		scaleinvrI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		scaleinvrI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		scaleinvrI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		scaleinvrU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		scaleinvrU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		scaleinvrU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		scaleinvrU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		scaleinvrU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		scaleinvrF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		scaleinvrF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		scaleinvrC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		scaleinvrC128(t.complex128s(), b)
	}
}

/* PowOf */

func (t *Dense) PowOf(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.powof(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.powof(other)
	case !safe:
		t.powof(other)
		retVal = t
	}
	return
}
func (t *Dense) powof(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		powofI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		powofI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		powofI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		powofI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		powofI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		powofU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		powofU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		powofU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		powofU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		powofU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		powofF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		powofF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		powofC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		powofC128(t.complex128s(), b)
	}
}

/* PowOfR */

func (t *Dense) PowOfR(other interface{}, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	switch {
	case incr:
	case toReuse:
		copyDense(reuse, t)
		reuse.powofr(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.powofr(other)
	case !safe:
		t.powofr(other)
		retVal = t
	}
	return
}
func (t *Dense) powofr(other interface{}) {
	switch t.t.Kind() {
	case reflect.Int:
		b := other.(int)
		powofrI(t.ints(), b)
	case reflect.Int8:
		b := other.(int8)
		powofrI8(t.int8s(), b)
	case reflect.Int16:
		b := other.(int16)
		powofrI16(t.int16s(), b)
	case reflect.Int32:
		b := other.(int32)
		powofrI32(t.int32s(), b)
	case reflect.Int64:
		b := other.(int64)
		powofrI64(t.int64s(), b)
	case reflect.Uint:
		b := other.(uint)
		powofrU(t.uints(), b)
	case reflect.Uint8:
		b := other.(uint8)
		powofrU8(t.uint8s(), b)
	case reflect.Uint16:
		b := other.(uint16)
		powofrU16(t.uint16s(), b)
	case reflect.Uint32:
		b := other.(uint32)
		powofrU32(t.uint32s(), b)
	case reflect.Uint64:
		b := other.(uint64)
		powofrU64(t.uint64s(), b)
	case reflect.Float32:
		b := other.(float32)
		powofrF32(t.float32s(), b)
	case reflect.Float64:
		b := other.(float64)
		powofrF64(t.float64s(), b)
	case reflect.Complex64:
		b := other.(complex64)
		powofrC64(t.complex64s(), b)
	case reflect.Complex128:
		b := other.(complex128)
		powofrC128(t.complex128s(), b)
	}
}
