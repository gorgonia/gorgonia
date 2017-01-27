package tensor

import (
	"math"
	"math/cmplx"
	"reflect"

	"github.com/chewxy/math32"
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
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.ints()
			for i := range data {
				data[i] += t.getI(i) + other.getI(i)
			}
		case reflect.Int8:
			data := reuse.int8s()
			for i := range data {
				data[i] += t.getI8(i) + other.getI8(i)
			}
		case reflect.Int16:
			data := reuse.int16s()
			for i := range data {
				data[i] += t.getI16(i) + other.getI16(i)
			}
		case reflect.Int32:
			data := reuse.int32s()
			for i := range data {
				data[i] += t.getI32(i) + other.getI32(i)
			}
		case reflect.Int64:
			data := reuse.int64s()
			for i := range data {
				data[i] += t.getI64(i) + other.getI64(i)
			}
		case reflect.Uint:
			data := reuse.uints()
			for i := range data {
				data[i] += t.getU(i) + other.getU(i)
			}
		case reflect.Uint8:
			data := reuse.uint8s()
			for i := range data {
				data[i] += t.getU8(i) + other.getU8(i)
			}
		case reflect.Uint16:
			data := reuse.uint16s()
			for i := range data {
				data[i] += t.getU16(i) + other.getU16(i)
			}
		case reflect.Uint32:
			data := reuse.uint32s()
			for i := range data {
				data[i] += t.getU32(i) + other.getU32(i)
			}
		case reflect.Uint64:
			data := reuse.uint64s()
			for i := range data {
				data[i] += t.getU64(i) + other.getU64(i)
			}
		case reflect.Float32:
			data := reuse.float32s()
			for i := range data {
				data[i] += t.getF32(i) + other.getF32(i)
			}
		case reflect.Float64:
			data := reuse.float64s()
			for i := range data {
				data[i] += t.getF64(i) + other.getF64(i)
			}
		case reflect.Complex64:
			data := reuse.complex64s()
			for i := range data {
				data[i] += t.getC64(i) + other.getC64(i)
			}
		case reflect.Complex128:
			data := reuse.complex128s()
			for i := range data {
				data[i] += t.getC128(i) + other.getC128(i)
			}
		}
		retVal = reuse
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
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.ints()
			for i := range data {
				data[i] += t.getI(i) - other.getI(i)
			}
		case reflect.Int8:
			data := reuse.int8s()
			for i := range data {
				data[i] += t.getI8(i) - other.getI8(i)
			}
		case reflect.Int16:
			data := reuse.int16s()
			for i := range data {
				data[i] += t.getI16(i) - other.getI16(i)
			}
		case reflect.Int32:
			data := reuse.int32s()
			for i := range data {
				data[i] += t.getI32(i) - other.getI32(i)
			}
		case reflect.Int64:
			data := reuse.int64s()
			for i := range data {
				data[i] += t.getI64(i) - other.getI64(i)
			}
		case reflect.Uint:
			data := reuse.uints()
			for i := range data {
				data[i] += t.getU(i) - other.getU(i)
			}
		case reflect.Uint8:
			data := reuse.uint8s()
			for i := range data {
				data[i] += t.getU8(i) - other.getU8(i)
			}
		case reflect.Uint16:
			data := reuse.uint16s()
			for i := range data {
				data[i] += t.getU16(i) - other.getU16(i)
			}
		case reflect.Uint32:
			data := reuse.uint32s()
			for i := range data {
				data[i] += t.getU32(i) - other.getU32(i)
			}
		case reflect.Uint64:
			data := reuse.uint64s()
			for i := range data {
				data[i] += t.getU64(i) - other.getU64(i)
			}
		case reflect.Float32:
			data := reuse.float32s()
			for i := range data {
				data[i] += t.getF32(i) - other.getF32(i)
			}
		case reflect.Float64:
			data := reuse.float64s()
			for i := range data {
				data[i] += t.getF64(i) - other.getF64(i)
			}
		case reflect.Complex64:
			data := reuse.complex64s()
			for i := range data {
				data[i] += t.getC64(i) - other.getC64(i)
			}
		case reflect.Complex128:
			data := reuse.complex128s()
			for i := range data {
				data[i] += t.getC128(i) - other.getC128(i)
			}
		}
		retVal = reuse
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
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.ints()
			for i := range data {
				data[i] += t.getI(i) * other.getI(i)
			}
		case reflect.Int8:
			data := reuse.int8s()
			for i := range data {
				data[i] += t.getI8(i) * other.getI8(i)
			}
		case reflect.Int16:
			data := reuse.int16s()
			for i := range data {
				data[i] += t.getI16(i) * other.getI16(i)
			}
		case reflect.Int32:
			data := reuse.int32s()
			for i := range data {
				data[i] += t.getI32(i) * other.getI32(i)
			}
		case reflect.Int64:
			data := reuse.int64s()
			for i := range data {
				data[i] += t.getI64(i) * other.getI64(i)
			}
		case reflect.Uint:
			data := reuse.uints()
			for i := range data {
				data[i] += t.getU(i) * other.getU(i)
			}
		case reflect.Uint8:
			data := reuse.uint8s()
			for i := range data {
				data[i] += t.getU8(i) * other.getU8(i)
			}
		case reflect.Uint16:
			data := reuse.uint16s()
			for i := range data {
				data[i] += t.getU16(i) * other.getU16(i)
			}
		case reflect.Uint32:
			data := reuse.uint32s()
			for i := range data {
				data[i] += t.getU32(i) * other.getU32(i)
			}
		case reflect.Uint64:
			data := reuse.uint64s()
			for i := range data {
				data[i] += t.getU64(i) * other.getU64(i)
			}
		case reflect.Float32:
			data := reuse.float32s()
			for i := range data {
				data[i] += t.getF32(i) * other.getF32(i)
			}
		case reflect.Float64:
			data := reuse.float64s()
			for i := range data {
				data[i] += t.getF64(i) * other.getF64(i)
			}
		case reflect.Complex64:
			data := reuse.complex64s()
			for i := range data {
				data[i] += t.getC64(i) * other.getC64(i)
			}
		case reflect.Complex128:
			data := reuse.complex128s()
			for i := range data {
				data[i] += t.getC128(i) * other.getC128(i)
			}
		}
		retVal = reuse
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

	var errs errorIndices
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.ints()
			for i := range data {
				if other.getI(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getI(i) / other.getI(i)
			}
		case reflect.Int8:
			data := reuse.int8s()
			for i := range data {
				if other.getI8(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getI8(i) / other.getI8(i)
			}
		case reflect.Int16:
			data := reuse.int16s()
			for i := range data {
				if other.getI16(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getI16(i) / other.getI16(i)
			}
		case reflect.Int32:
			data := reuse.int32s()
			for i := range data {
				if other.getI32(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getI32(i) / other.getI32(i)
			}
		case reflect.Int64:
			data := reuse.int64s()
			for i := range data {
				if other.getI64(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getI64(i) / other.getI64(i)
			}
		case reflect.Uint:
			data := reuse.uints()
			for i := range data {
				if other.getU(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getU(i) / other.getU(i)
			}
		case reflect.Uint8:
			data := reuse.uint8s()
			for i := range data {
				if other.getU8(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getU8(i) / other.getU8(i)
			}
		case reflect.Uint16:
			data := reuse.uint16s()
			for i := range data {
				if other.getU16(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getU16(i) / other.getU16(i)
			}
		case reflect.Uint32:
			data := reuse.uint32s()
			for i := range data {
				if other.getU32(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getU32(i) / other.getU32(i)
			}
		case reflect.Uint64:
			data := reuse.uint64s()
			for i := range data {
				if other.getU64(i) == 0 {
					errs = append(errs, i)
					continue
				}
				data[i] += t.getU64(i) / other.getU64(i)
			}
		case reflect.Float32:
			data := reuse.float32s()
			for i := range data {
				data[i] += t.getF32(i) / other.getF32(i)
			}
		case reflect.Float64:
			data := reuse.float64s()
			for i := range data {
				data[i] += t.getF64(i) / other.getF64(i)
			}
		case reflect.Complex64:
			data := reuse.complex64s()
			for i := range data {
				data[i] += t.getC64(i) / other.getC64(i)
			}
		case reflect.Complex128:
			data := reuse.complex128s()
			for i := range data {
				data[i] += t.getC128(i) / other.getC128(i)
			}
		}
		if errs != nil {
			err = err
		}
		retVal = reuse
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
		switch reuse.t.Kind() {
		case reflect.Int:
			data := reuse.ints()
			for i := range data {
				data[i] += int(math.Pow(float64(t.getI(i)), float64(other.getI(i))))
			}
		case reflect.Int8:
			data := reuse.int8s()
			for i := range data {
				data[i] += int8(math.Pow(float64(t.getI8(i)), float64(other.getI8(i))))
			}
		case reflect.Int16:
			data := reuse.int16s()
			for i := range data {
				data[i] += int16(math.Pow(float64(t.getI16(i)), float64(other.getI16(i))))
			}
		case reflect.Int32:
			data := reuse.int32s()
			for i := range data {
				data[i] += int32(math.Pow(float64(t.getI32(i)), float64(other.getI32(i))))
			}
		case reflect.Int64:
			data := reuse.int64s()
			for i := range data {
				data[i] += int64(math.Pow(float64(t.getI64(i)), float64(other.getI64(i))))
			}
		case reflect.Uint:
			data := reuse.uints()
			for i := range data {
				data[i] += uint(math.Pow(float64(t.getU(i)), float64(other.getU(i))))
			}
		case reflect.Uint8:
			data := reuse.uint8s()
			for i := range data {
				data[i] += uint8(math.Pow(float64(t.getU8(i)), float64(other.getU8(i))))
			}
		case reflect.Uint16:
			data := reuse.uint16s()
			for i := range data {
				data[i] += uint16(math.Pow(float64(t.getU16(i)), float64(other.getU16(i))))
			}
		case reflect.Uint32:
			data := reuse.uint32s()
			for i := range data {
				data[i] += uint32(math.Pow(float64(t.getU32(i)), float64(other.getU32(i))))
			}
		case reflect.Uint64:
			data := reuse.uint64s()
			for i := range data {
				data[i] += uint64(math.Pow(float64(t.getU64(i)), float64(other.getU64(i))))
			}
		case reflect.Float32:
			data := reuse.float32s()
			for i := range data {
				data[i] +=
					math32.Pow(t.getF32(i), other.getF32(i))
			}
		case reflect.Float64:
			data := reuse.float64s()
			for i := range data {
				data[i] += math.Pow(t.getF64(i), other.getF64(i))
			}
		case reflect.Complex64:
			data := reuse.complex64s()
			for i := range data {
				data[i] += complex64(cmplx.Pow(complex128(t.getC64(i)), complex128(other.getC64(i))))
			}
		case reflect.Complex128:
			data := reuse.complex128s()
			for i := range data {
				data[i] += cmplx.Pow(t.getC128(i), other.getC128(i))
			}
		}
		retVal = reuse
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrTransI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrTransI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrTransI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrTransI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrTransI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrTransU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrTransU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrTransU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrTransU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrTransU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrTransF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrTransF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrTransC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrTransC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrTransInvI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrTransInvI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrTransInvI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrTransInvI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrTransInvI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrTransInvU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrTransInvU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrTransInvU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrTransInvU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrTransInvU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrTransInvF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrTransInvF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrTransInvC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrTransInvC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrTransInvRI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrTransInvRI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrTransInvRI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrTransInvRI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrTransInvRI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrTransInvRU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrTransInvRU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrTransInvRU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrTransInvRU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrTransInvRU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrTransInvRF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrTransInvRF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrTransInvRC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrTransInvRC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrScaleI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrScaleI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrScaleI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrScaleI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrScaleI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrScaleU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrScaleU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrScaleU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrScaleU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrScaleU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrScaleF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrScaleF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrScaleC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrScaleC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrScaleInvI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrScaleInvI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrScaleInvI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrScaleInvI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrScaleInvI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrScaleInvU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrScaleInvU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrScaleInvU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrScaleInvU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrScaleInvU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrScaleInvF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrScaleInvF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrScaleInvC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrScaleInvC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrScaleInvRI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrScaleInvRI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrScaleInvRI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrScaleInvRI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrScaleInvRI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrScaleInvRU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrScaleInvRU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrScaleInvRU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrScaleInvRU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrScaleInvRU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrScaleInvRF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrScaleInvRF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrScaleInvRC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrScaleInvRC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrPowOfI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrPowOfI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrPowOfI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrPowOfI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrPowOfI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrPowOfU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrPowOfU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrPowOfU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrPowOfU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrPowOfU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrPowOfF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrPowOfF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrPowOfC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrPowOfC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
		switch t.t.Kind() {
		case reflect.Int:
			err = incrPowOfRI(t.ints(), reuse.ints(), other.(int))
			retVal = reuse
		case reflect.Int8:
			err = incrPowOfRI8(t.int8s(), reuse.int8s(), other.(int8))
			retVal = reuse
		case reflect.Int16:
			err = incrPowOfRI16(t.int16s(), reuse.int16s(), other.(int16))
			retVal = reuse
		case reflect.Int32:
			err = incrPowOfRI32(t.int32s(), reuse.int32s(), other.(int32))
			retVal = reuse
		case reflect.Int64:
			err = incrPowOfRI64(t.int64s(), reuse.int64s(), other.(int64))
			retVal = reuse
		case reflect.Uint:
			err = incrPowOfRU(t.uints(), reuse.uints(), other.(uint))
			retVal = reuse
		case reflect.Uint8:
			err = incrPowOfRU8(t.uint8s(), reuse.uint8s(), other.(uint8))
			retVal = reuse
		case reflect.Uint16:
			err = incrPowOfRU16(t.uint16s(), reuse.uint16s(), other.(uint16))
			retVal = reuse
		case reflect.Uint32:
			err = incrPowOfRU32(t.uint32s(), reuse.uint32s(), other.(uint32))
			retVal = reuse
		case reflect.Uint64:
			err = incrPowOfRU64(t.uint64s(), reuse.uint64s(), other.(uint64))
			retVal = reuse
		case reflect.Float32:
			err = incrPowOfRF32(t.float32s(), reuse.float32s(), other.(float32))
			retVal = reuse
		case reflect.Float64:
			err = incrPowOfRF64(t.float64s(), reuse.float64s(), other.(float64))
			retVal = reuse
		case reflect.Complex64:
			err = incrPowOfRC64(t.complex64s(), reuse.complex64s(), other.(complex64))
			retVal = reuse
		case reflect.Complex128:
			err = incrPowOfRC128(t.complex128s(), reuse.complex128s(), other.(complex128))
			retVal = reuse
		}
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
