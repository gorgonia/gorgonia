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

		if reuse.len() != a.Shape().TotalSize() {
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

		if reuse.len() != a.Shape().TotalSize() {
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

	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
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
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.add(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.add(other, nil, ot)
	case !safe:
		err = t.add(other, it, ot)
		retVal = t
	}
	return
}
func (t *Dense) add(other *Dense, it, ot *FlatIterator) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.ints()
		odata := other.ints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddI(tdata, odata)
		}
	case reflect.Int8:
		tdata := t.int8s()
		odata := other.int8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddI8(tdata, odata)
		}
	case reflect.Int16:
		tdata := t.int16s()
		odata := other.int16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddI16(tdata, odata)
		}
	case reflect.Int32:
		tdata := t.int32s()
		odata := other.int32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddI32(tdata, odata)
		}
	case reflect.Int64:
		tdata := t.int64s()
		odata := other.int64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddI64(tdata, odata)
		}
	case reflect.Uint:
		tdata := t.uints()
		odata := other.uints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddU(tdata, odata)
		}
	case reflect.Uint8:
		tdata := t.uint8s()
		odata := other.uint8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddU8(tdata, odata)
		}
	case reflect.Uint16:
		tdata := t.uint16s()
		odata := other.uint16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddU16(tdata, odata)
		}
	case reflect.Uint32:
		tdata := t.uint32s()
		odata := other.uint32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddU32(tdata, odata)
		}
	case reflect.Uint64:
		tdata := t.uint64s()
		odata := other.uint64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddU64(tdata, odata)
		}
	case reflect.Float32:
		tdata := t.float32s()
		odata := other.float32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddF32(tdata, odata)
		}
	case reflect.Float64:
		tdata := t.float64s()
		odata := other.float64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddF64(tdata, odata)
		}
	case reflect.Complex64:
		tdata := t.complex64s()
		odata := other.complex64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddC64(tdata, odata)
		}
	case reflect.Complex128:
		tdata := t.complex128s()
		odata := other.complex128s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] + odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] + odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] + odata[j]
				i++
			}
		default:
			vecAddC128(tdata, odata)
		}
	default:
		// TODO: Handle Number interface
	}
	return nil
}

/* Sub */

func (t *Dense) Sub(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
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
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.sub(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.sub(other, nil, ot)
	case !safe:
		err = t.sub(other, it, ot)
		retVal = t
	}
	return
}
func (t *Dense) sub(other *Dense, it, ot *FlatIterator) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.ints()
		odata := other.ints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubI(tdata, odata)
		}
	case reflect.Int8:
		tdata := t.int8s()
		odata := other.int8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubI8(tdata, odata)
		}
	case reflect.Int16:
		tdata := t.int16s()
		odata := other.int16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubI16(tdata, odata)
		}
	case reflect.Int32:
		tdata := t.int32s()
		odata := other.int32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubI32(tdata, odata)
		}
	case reflect.Int64:
		tdata := t.int64s()
		odata := other.int64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubI64(tdata, odata)
		}
	case reflect.Uint:
		tdata := t.uints()
		odata := other.uints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubU(tdata, odata)
		}
	case reflect.Uint8:
		tdata := t.uint8s()
		odata := other.uint8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubU8(tdata, odata)
		}
	case reflect.Uint16:
		tdata := t.uint16s()
		odata := other.uint16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubU16(tdata, odata)
		}
	case reflect.Uint32:
		tdata := t.uint32s()
		odata := other.uint32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubU32(tdata, odata)
		}
	case reflect.Uint64:
		tdata := t.uint64s()
		odata := other.uint64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubU64(tdata, odata)
		}
	case reflect.Float32:
		tdata := t.float32s()
		odata := other.float32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubF32(tdata, odata)
		}
	case reflect.Float64:
		tdata := t.float64s()
		odata := other.float64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubF64(tdata, odata)
		}
	case reflect.Complex64:
		tdata := t.complex64s()
		odata := other.complex64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubC64(tdata, odata)
		}
	case reflect.Complex128:
		tdata := t.complex128s()
		odata := other.complex128s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] - odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] - odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] - odata[j]
				i++
			}
		default:
			vecSubC128(tdata, odata)
		}
	default:
		// TODO: Handle Number interface
	}
	return nil
}

/* Mul */

func (t *Dense) Mul(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
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
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.mul(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.mul(other, nil, ot)
	case !safe:
		err = t.mul(other, it, ot)
		retVal = t
	}
	return
}
func (t *Dense) mul(other *Dense, it, ot *FlatIterator) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.ints()
		odata := other.ints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulI(tdata, odata)
		}
	case reflect.Int8:
		tdata := t.int8s()
		odata := other.int8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulI8(tdata, odata)
		}
	case reflect.Int16:
		tdata := t.int16s()
		odata := other.int16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulI16(tdata, odata)
		}
	case reflect.Int32:
		tdata := t.int32s()
		odata := other.int32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulI32(tdata, odata)
		}
	case reflect.Int64:
		tdata := t.int64s()
		odata := other.int64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulI64(tdata, odata)
		}
	case reflect.Uint:
		tdata := t.uints()
		odata := other.uints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulU(tdata, odata)
		}
	case reflect.Uint8:
		tdata := t.uint8s()
		odata := other.uint8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulU8(tdata, odata)
		}
	case reflect.Uint16:
		tdata := t.uint16s()
		odata := other.uint16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulU16(tdata, odata)
		}
	case reflect.Uint32:
		tdata := t.uint32s()
		odata := other.uint32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulU32(tdata, odata)
		}
	case reflect.Uint64:
		tdata := t.uint64s()
		odata := other.uint64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulU64(tdata, odata)
		}
	case reflect.Float32:
		tdata := t.float32s()
		odata := other.float32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulF32(tdata, odata)
		}
	case reflect.Float64:
		tdata := t.float64s()
		odata := other.float64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulF64(tdata, odata)
		}
	case reflect.Complex64:
		tdata := t.complex64s()
		odata := other.complex64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulC64(tdata, odata)
		}
	case reflect.Complex128:
		tdata := t.complex128s()
		odata := other.complex128s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] * odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] * odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] * odata[j]
				i++
			}
		default:
			vecMulC128(tdata, odata)
		}
	default:
		// TODO: Handle Number interface
	}
	return nil
}

/* Div */

func (t *Dense) Div(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	var errs errorIndices
	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
	}
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
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.div(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.div(other, nil, ot)
	case !safe:
		err = t.div(other, it, ot)
		retVal = t
	}
	return
}
func (t *Dense) div(other *Dense, it, ot *FlatIterator) (err error) {
	var errs errorIndices
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.ints()
		odata := other.ints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivI(tdata, odata)
		}
	case reflect.Int8:
		tdata := t.int8s()
		odata := other.int8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivI8(tdata, odata)
		}
	case reflect.Int16:
		tdata := t.int16s()
		odata := other.int16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivI16(tdata, odata)
		}
	case reflect.Int32:
		tdata := t.int32s()
		odata := other.int32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivI32(tdata, odata)
		}
	case reflect.Int64:
		tdata := t.int64s()
		odata := other.int64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivI64(tdata, odata)
		}
	case reflect.Uint:
		tdata := t.uints()
		odata := other.uints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivU(tdata, odata)
		}
	case reflect.Uint8:
		tdata := t.uint8s()
		odata := other.uint8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivU8(tdata, odata)
		}
	case reflect.Uint16:
		tdata := t.uint16s()
		odata := other.uint16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivU16(tdata, odata)
		}
	case reflect.Uint32:
		tdata := t.uint32s()
		odata := other.uint32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivU32(tdata, odata)
		}
	case reflect.Uint64:
		tdata := t.uint64s()
		odata := other.uint64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				if odata[j] == 0 {
					errs = append(errs, j)
					continue
				}
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivU64(tdata, odata)
		}
	case reflect.Float32:
		tdata := t.float32s()
		odata := other.float32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivF32(tdata, odata)
		}
	case reflect.Float64:
		tdata := t.float64s()
		odata := other.float64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivF64(tdata, odata)
		}
	case reflect.Complex64:
		tdata := t.complex64s()
		odata := other.complex64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivC64(tdata, odata)
		}
	case reflect.Complex128:
		tdata := t.complex128s()
		odata := other.complex128s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = tdata[i] / odata[j]
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = tdata[i] / odata[j]
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = tdata[i] / odata[j]
				i++
			}
		default:
			vecDivC128(tdata, odata)
		}
	default:
		// TODO: Handle Number interface
	}
	if errs != nil {
		err = err
	}
	return nil
}

/* Pow */

func (t *Dense) Pow(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
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
				data[i] += math32.Pow(t.getF32(i), other.getF32(i))
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
		if t.IsMaterializable() {
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.pow(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable() {
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.pow(other, nil, ot)
	case !safe:
		err = t.pow(other, it, ot)
		retVal = t
	}
	return
}
func (t *Dense) pow(other *Dense, it, ot *FlatIterator) (err error) {
	switch t.t.Kind() {
	case reflect.Int:
		tdata := t.ints()
		odata := other.ints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = int(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowI(tdata, odata)
		}
	case reflect.Int8:
		tdata := t.int8s()
		odata := other.int8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = int8(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowI8(tdata, odata)
		}
	case reflect.Int16:
		tdata := t.int16s()
		odata := other.int16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = int16(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowI16(tdata, odata)
		}
	case reflect.Int32:
		tdata := t.int32s()
		odata := other.int32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = int32(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowI32(tdata, odata)
		}
	case reflect.Int64:
		tdata := t.int64s()
		odata := other.int64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = int64(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowI64(tdata, odata)
		}
	case reflect.Uint:
		tdata := t.uints()
		odata := other.uints()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = uint(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowU(tdata, odata)
		}
	case reflect.Uint8:
		tdata := t.uint8s()
		odata := other.uint8s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = uint8(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowU8(tdata, odata)
		}
	case reflect.Uint16:
		tdata := t.uint16s()
		odata := other.uint16s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = uint16(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowU16(tdata, odata)
		}
	case reflect.Uint32:
		tdata := t.uint32s()
		odata := other.uint32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = uint32(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowU32(tdata, odata)
		}
	case reflect.Uint64:
		tdata := t.uint64s()
		odata := other.uint64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = uint64(math.Pow(float64(tdata[i]), float64(odata[j])))
				i++
			}
		default:
			vecPowU64(tdata, odata)
		}
	case reflect.Float32:
		tdata := t.float32s()
		odata := other.float32s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = math32.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = math32.Pow(tdata[i], odata[j])
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = math32.Pow(tdata[i], odata[j])
				i++
			}
		default:
			vecPowF32(tdata, odata)
		}
	case reflect.Float64:
		tdata := t.float64s()
		odata := other.float64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = math.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = math.Pow(tdata[i], odata[j])
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = math.Pow(tdata[i], odata[j])
				i++
			}
		default:
			vecPowF64(tdata, odata)
		}
	case reflect.Complex64:
		tdata := t.complex64s()
		odata := other.complex64s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
				i++
			}
		default:
			vecPowC64(tdata, odata)
		}
	case reflect.Complex128:
		tdata := t.complex128s()
		odata := other.complex128s()
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				tdata[i] = cmplx.Pow(tdata[i], odata[j])
				i++
			}
		default:
			vecPowC128(tdata, odata)
		}
	default:
		// TODO: Handle Number interface
	}
	return nil
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
