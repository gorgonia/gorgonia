package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

// Sum returns the sum of the elements of the tensor along the given axes.
// If multiple axes are given then this method will return the sum of the Tensor according the the order of the axes provided
func (t *Dense) Sum(along ...int) (retVal *Dense, err error) {
	monotonic, incr1 := IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		var ret interface{}
		switch t.t.Kind() {
		case reflect.Int:
			ret = sumI(t.ints())
		case reflect.Int8:
			ret = sumI8(t.int8s())
		case reflect.Int16:
			ret = sumI16(t.int16s())
		case reflect.Int32:
			ret = sumI32(t.int32s())
		case reflect.Int64:
			ret = sumI64(t.int64s())
		case reflect.Uint:
			ret = sumU(t.uints())
		case reflect.Uint8:
			ret = sumU8(t.uint8s())
		case reflect.Uint16:
			ret = sumU16(t.uint16s())
		case reflect.Uint32:
			ret = sumU32(t.uint32s())
		case reflect.Uint64:
			ret = sumU64(t.uint64s())
		case reflect.Float32:
			ret = sumF32(t.float32s())
		case reflect.Float64:
			ret = sumF64(t.float64s())
		case reflect.Complex64:
			ret = sumC64(t.complex64s())
		case reflect.Complex128:
			ret = sumC128(t.complex128s())
		}
		retVal = New(FromScalar(ret))
		return
	}
	retVal = t
	prev := -1
	dims := len(retVal.Shape())

	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
			return
		}

		retVal = retVal.sum(axis)
	}
	return
}
func (t *Dense) sum(axis int) (retVal *Dense) {
	switch t.t.Kind() {
	case reflect.Int:
		return t.sReduceI(axis, vecAddI, sumI, addI)
	case reflect.Int8:
		return t.sReduceI8(axis, vecAddI8, sumI8, addI8)
	case reflect.Int16:
		return t.sReduceI16(axis, vecAddI16, sumI16, addI16)
	case reflect.Int32:
		return t.sReduceI32(axis, vecAddI32, sumI32, addI32)
	case reflect.Int64:
		return t.sReduceI64(axis, vecAddI64, sumI64, addI64)
	case reflect.Uint:
		return t.sReduceU(axis, vecAddU, sumU, addU)
	case reflect.Uint8:
		return t.sReduceU8(axis, vecAddU8, sumU8, addU8)
	case reflect.Uint16:
		return t.sReduceU16(axis, vecAddU16, sumU16, addU16)
	case reflect.Uint32:
		return t.sReduceU32(axis, vecAddU32, sumU32, addU32)
	case reflect.Uint64:
		return t.sReduceU64(axis, vecAddU64, sumU64, addU64)
	case reflect.Float32:
		return t.sReduceF32(axis, vecAddF32, sumF32, addF32)
	case reflect.Float64:
		return t.sReduceF64(axis, vecAddF64, sumF64, addF64)
	case reflect.Complex64:
		return t.sReduceC64(axis, vecAddC64, sumC64, addC64)
	case reflect.Complex128:
		return t.sReduceC128(axis, vecAddC128, sumC128, addC128)
	}
	panic("Unreachable")
}
