package tensor

import (
	"reflect"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Eq */

func (t *Dense) eqDD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}

	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		ret := eqDDBoolsB(td, od)
		retVal.fromSlice(ret)
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		if same {
			ret := eqDDSameI(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsI(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		if same {
			ret := eqDDSameI8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsI8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		if same {
			ret := eqDDSameI16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsI16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		if same {
			ret := eqDDSameI32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsI32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		if same {
			ret := eqDDSameI64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsI64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		if same {
			ret := eqDDSameU(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsU(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		if same {
			ret := eqDDSameU8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsU8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		if same {
			ret := eqDDSameU16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsU16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		if same {
			ret := eqDDSameU32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsU32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		if same {
			ret := eqDDSameU64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsU64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		ret := eqDDBoolsUintptr(td, od)
		retVal.fromSlice(ret)
	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		if same {
			ret := eqDDSameF32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsF32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		if same {
			ret := eqDDSameF64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsF64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		if same {
			ret := eqDDSameC64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsC64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		if same {
			ret := eqDDSameC128(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := eqDDBoolsC128(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.String:
		td := t.strings()
		od := other.strings()
		ret := eqDDBoolsStr(td, od)
		retVal.fromSlice(ret)
	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		ret := eqDDBoolsUnsafePointer(td, od)
		retVal.fromSlice(ret)
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "eq")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}

/* Gt */

func (t *Dense) gtDD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}

	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		ret := gtDDBoolsB(td, od)
		retVal.fromSlice(ret)
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		if same {
			ret := gtDDSameI(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsI(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		if same {
			ret := gtDDSameI8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsI8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		if same {
			ret := gtDDSameI16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsI16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		if same {
			ret := gtDDSameI32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsI32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		if same {
			ret := gtDDSameI64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsI64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		if same {
			ret := gtDDSameU(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsU(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		if same {
			ret := gtDDSameU8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsU8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		if same {
			ret := gtDDSameU16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsU16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		if same {
			ret := gtDDSameU32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsU32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		if same {
			ret := gtDDSameU64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsU64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		ret := gtDDBoolsUintptr(td, od)
		retVal.fromSlice(ret)
	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		if same {
			ret := gtDDSameF32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsF32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		if same {
			ret := gtDDSameF64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsF64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		if same {
			ret := gtDDSameC64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsC64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		if same {
			ret := gtDDSameC128(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gtDDBoolsC128(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.String:
		td := t.strings()
		od := other.strings()
		ret := gtDDBoolsStr(td, od)
		retVal.fromSlice(ret)
	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		ret := gtDDBoolsUnsafePointer(td, od)
		retVal.fromSlice(ret)
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "gt")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}

/* Gte */

func (t *Dense) gteDD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}

	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		ret := gteDDBoolsB(td, od)
		retVal.fromSlice(ret)
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		if same {
			ret := gteDDSameI(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsI(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		if same {
			ret := gteDDSameI8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsI8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		if same {
			ret := gteDDSameI16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsI16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		if same {
			ret := gteDDSameI32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsI32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		if same {
			ret := gteDDSameI64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsI64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		if same {
			ret := gteDDSameU(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsU(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		if same {
			ret := gteDDSameU8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsU8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		if same {
			ret := gteDDSameU16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsU16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		if same {
			ret := gteDDSameU32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsU32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		if same {
			ret := gteDDSameU64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsU64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		ret := gteDDBoolsUintptr(td, od)
		retVal.fromSlice(ret)
	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		if same {
			ret := gteDDSameF32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsF32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		if same {
			ret := gteDDSameF64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsF64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		if same {
			ret := gteDDSameC64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsC64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		if same {
			ret := gteDDSameC128(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := gteDDBoolsC128(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.String:
		td := t.strings()
		od := other.strings()
		ret := gteDDBoolsStr(td, od)
		retVal.fromSlice(ret)
	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		ret := gteDDBoolsUnsafePointer(td, od)
		retVal.fromSlice(ret)
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "gte")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}

/* Lt */

func (t *Dense) ltDD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}

	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		ret := ltDDBoolsB(td, od)
		retVal.fromSlice(ret)
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		if same {
			ret := ltDDSameI(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsI(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		if same {
			ret := ltDDSameI8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsI8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		if same {
			ret := ltDDSameI16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsI16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		if same {
			ret := ltDDSameI32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsI32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		if same {
			ret := ltDDSameI64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsI64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		if same {
			ret := ltDDSameU(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsU(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		if same {
			ret := ltDDSameU8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsU8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		if same {
			ret := ltDDSameU16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsU16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		if same {
			ret := ltDDSameU32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsU32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		if same {
			ret := ltDDSameU64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsU64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		ret := ltDDBoolsUintptr(td, od)
		retVal.fromSlice(ret)
	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		if same {
			ret := ltDDSameF32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsF32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		if same {
			ret := ltDDSameF64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsF64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		if same {
			ret := ltDDSameC64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsC64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		if same {
			ret := ltDDSameC128(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := ltDDBoolsC128(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.String:
		td := t.strings()
		od := other.strings()
		ret := ltDDBoolsStr(td, od)
		retVal.fromSlice(ret)
	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		ret := ltDDBoolsUnsafePointer(td, od)
		retVal.fromSlice(ret)
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "lt")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}

/* Lte */

func (t *Dense) lteDD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}

	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	case reflect.Bool:
		td := t.bools()
		od := other.bools()
		ret := lteDDBoolsB(td, od)
		retVal.fromSlice(ret)
	case reflect.Int:
		td := t.ints()
		od := other.ints()
		if same {
			ret := lteDDSameI(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsI(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int8:
		td := t.int8s()
		od := other.int8s()
		if same {
			ret := lteDDSameI8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsI8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int16:
		td := t.int16s()
		od := other.int16s()
		if same {
			ret := lteDDSameI16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsI16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int32:
		td := t.int32s()
		od := other.int32s()
		if same {
			ret := lteDDSameI32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsI32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Int64:
		td := t.int64s()
		od := other.int64s()
		if same {
			ret := lteDDSameI64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsI64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint:
		td := t.uints()
		od := other.uints()
		if same {
			ret := lteDDSameU(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsU(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint8:
		td := t.uint8s()
		od := other.uint8s()
		if same {
			ret := lteDDSameU8(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsU8(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint16:
		td := t.uint16s()
		od := other.uint16s()
		if same {
			ret := lteDDSameU16(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsU16(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint32:
		td := t.uint32s()
		od := other.uint32s()
		if same {
			ret := lteDDSameU32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsU32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uint64:
		td := t.uint64s()
		od := other.uint64s()
		if same {
			ret := lteDDSameU64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsU64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Uintptr:
		td := t.uintptrs()
		od := other.uintptrs()
		ret := lteDDBoolsUintptr(td, od)
		retVal.fromSlice(ret)
	case reflect.Float32:
		td := t.float32s()
		od := other.float32s()
		if same {
			ret := lteDDSameF32(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsF32(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Float64:
		td := t.float64s()
		od := other.float64s()
		if same {
			ret := lteDDSameF64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsF64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex64:
		td := t.complex64s()
		od := other.complex64s()
		if same {
			ret := lteDDSameC64(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsC64(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.Complex128:
		td := t.complex128s()
		od := other.complex128s()
		if same {
			ret := lteDDSameC128(td, od)
			retVal.fromSlice(ret)
		} else {
			ret := lteDDBoolsC128(td, od)
			retVal.fromSlice(ret)
		}
	case reflect.String:
		td := t.strings()
		od := other.strings()
		ret := lteDDBoolsStr(td, od)
		retVal.fromSlice(ret)
	case reflect.UnsafePointer:
		td := t.unsafePointers()
		od := other.unsafePointers()
		ret := lteDDBoolsUnsafePointer(td, od)
		retVal.fromSlice(ret)
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "lte")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}
