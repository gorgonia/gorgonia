package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

// Set sets the value of the underlying array at the index i.
func (a *array) Set(i int, x interface{}) {
	switch a.t.Kind() {
	case reflect.Bool:
		xv := x.(bool)
		a.SetB(i, xv)
	case reflect.Int:
		xv := x.(int)
		a.SetI(i, xv)
	case reflect.Int8:
		xv := x.(int8)
		a.SetI8(i, xv)
	case reflect.Int16:
		xv := x.(int16)
		a.SetI16(i, xv)
	case reflect.Int32:
		xv := x.(int32)
		a.SetI32(i, xv)
	case reflect.Int64:
		xv := x.(int64)
		a.SetI64(i, xv)
	case reflect.Uint:
		xv := x.(uint)
		a.SetU(i, xv)
	case reflect.Uint8:
		xv := x.(uint8)
		a.SetU8(i, xv)
	case reflect.Uint16:
		xv := x.(uint16)
		a.SetU16(i, xv)
	case reflect.Uint32:
		xv := x.(uint32)
		a.SetU32(i, xv)
	case reflect.Uint64:
		xv := x.(uint64)
		a.SetU64(i, xv)
	case reflect.Uintptr:
		xv := x.(uintptr)
		a.SetUintptr(i, xv)
	case reflect.Float32:
		xv := x.(float32)
		a.SetF32(i, xv)
	case reflect.Float64:
		xv := x.(float64)
		a.SetF64(i, xv)
	case reflect.Complex64:
		xv := x.(complex64)
		a.SetC64(i, xv)
	case reflect.Complex128:
		xv := x.(complex128)
		a.SetC128(i, xv)
	case reflect.String:
		xv := x.(string)
		a.SetStr(i, xv)
	case reflect.UnsafePointer:
		xv := x.(unsafe.Pointer)
		a.SetUnsafePointer(i, xv)
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.Ptr)
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

// Get returns the ith element of the underlying array of the *Dense tensor.
func (a *array) Get(i int) interface{} {
	switch a.t.Kind() {
	case reflect.Bool:
		return a.GetB(i)
	case reflect.Int:
		return a.GetI(i)
	case reflect.Int8:
		return a.GetI8(i)
	case reflect.Int16:
		return a.GetI16(i)
	case reflect.Int32:
		return a.GetI32(i)
	case reflect.Int64:
		return a.GetI64(i)
	case reflect.Uint:
		return a.GetU(i)
	case reflect.Uint8:
		return a.GetU8(i)
	case reflect.Uint16:
		return a.GetU16(i)
	case reflect.Uint32:
		return a.GetU32(i)
	case reflect.Uint64:
		return a.GetU64(i)
	case reflect.Uintptr:
		return a.GetUintptr(i)
	case reflect.Float32:
		return a.GetF32(i)
	case reflect.Float64:
		return a.GetF64(i)
	case reflect.Complex64:
		return a.GetC64(i)
	case reflect.Complex128:
		return a.GetC128(i)
	case reflect.String:
		return a.GetStr(i)
	case reflect.UnsafePointer:
		return a.GetUnsafePointer(i)
	default:
		at := uintptr(a.Ptr) + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(at))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

// Memset sets all values in the array.
func (a *array) Memset(x interface{}) error {
	switch a.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Bools()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Ints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Int64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uint64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Uintptrs()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Float32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Float64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Complex64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Complex128s()
		for i := range data {
			data[i] = xv
		}
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.Strings()
		for i := range data {
			data[i] = xv
		}
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, a.t, x)
		}
		data := a.UnsafePointers()
		for i := range data {
			data[i] = xv
		}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.Ptr)
		for i := 0; i < a.L; i++ {
			want := ptr + uintptr(i)*a.t.Size()
			val := reflect.NewAt(a.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
	}
	return nil
}

// Eq checks that any two arrays are equal
func (a array) Eq(other interface{}) bool {
	if oa, ok := other.(array); ok {
		if oa.t != a.t {
			return false
		}

		if oa.L != a.L {
			return false
		}

		if oa.C != a.C {
			return false
		}

		// same exact thing
		if uintptr(oa.Ptr) == uintptr(a.Ptr) {
			return true
		}

		switch a.t.Kind() {
		case reflect.Bool:
			for i, v := range a.Bools() {
				if oa.GetB(i) != v {
					return false
				}
			}
		case reflect.Int:
			for i, v := range a.Ints() {
				if oa.GetI(i) != v {
					return false
				}
			}
		case reflect.Int8:
			for i, v := range a.Int8s() {
				if oa.GetI8(i) != v {
					return false
				}
			}
		case reflect.Int16:
			for i, v := range a.Int16s() {
				if oa.GetI16(i) != v {
					return false
				}
			}
		case reflect.Int32:
			for i, v := range a.Int32s() {
				if oa.GetI32(i) != v {
					return false
				}
			}
		case reflect.Int64:
			for i, v := range a.Int64s() {
				if oa.GetI64(i) != v {
					return false
				}
			}
		case reflect.Uint:
			for i, v := range a.Uints() {
				if oa.GetU(i) != v {
					return false
				}
			}
		case reflect.Uint8:
			for i, v := range a.Uint8s() {
				if oa.GetU8(i) != v {
					return false
				}
			}
		case reflect.Uint16:
			for i, v := range a.Uint16s() {
				if oa.GetU16(i) != v {
					return false
				}
			}
		case reflect.Uint32:
			for i, v := range a.Uint32s() {
				if oa.GetU32(i) != v {
					return false
				}
			}
		case reflect.Uint64:
			for i, v := range a.Uint64s() {
				if oa.GetU64(i) != v {
					return false
				}
			}
		case reflect.Uintptr:
			for i, v := range a.Uintptrs() {
				if oa.GetUintptr(i) != v {
					return false
				}
			}
		case reflect.Float32:
			for i, v := range a.Float32s() {
				if oa.GetF32(i) != v {
					return false
				}
			}
		case reflect.Float64:
			for i, v := range a.Float64s() {
				if oa.GetF64(i) != v {
					return false
				}
			}
		case reflect.Complex64:
			for i, v := range a.Complex64s() {
				if oa.GetC64(i) != v {
					return false
				}
			}
		case reflect.Complex128:
			for i, v := range a.Complex128s() {
				if oa.GetC128(i) != v {
					return false
				}
			}
		case reflect.String:
			for i, v := range a.Strings() {
				if oa.GetStr(i) != v {
					return false
				}
			}
		case reflect.UnsafePointer:
			for i, v := range a.UnsafePointers() {
				if oa.GetUnsafePointer(i) != v {
					return false
				}
			}
		default:
			for i := 0; i < a.L; i++ {
				if !reflect.DeepEqual(a.Get(i), oa.Get(i)) {
					return false
				}
			}
		}
		return true
	}
	return false
}
