package tensor

import (
	"unsafe"
)

// FillValue returns the value used to fill the invalid entries of a masked array
func (t *Dense) FillValue() interface{} {
	switch t.Dtype() {
	case Bool:
		return true
	case Int:
		return int(999999)
	case Int8:
		return int8(99)
	case Int16:
		return int16(9999)
	case Int32:
		return int32(999999)
	case Int64:
		return int64(999999)
	case Uint:
		return uint(999999)
	case Byte:
		return byte(99)
	case Uint8:
		return uint8(99)
	case Uint16:
		return uint16(9999)
	case Uint32:
		return uint32(999999)
	case Uint64:
		return uint64(999999)
	case Float32:
		return float32(1.0e20)
	case Float64:
		return float64(1.0e20)
	case Complex64:
		return complex64(1.0e20 + 0i)
	case Complex128:
		return complex128(1.0e20 + 0i)
	case String:
		return `N/A`
	case Uintptr:
		return uintptr(0x999999)
	case UnsafePointer:
		return unsafe.Pointer(nil)
	default:
		return nil
	}
}

// Filled returns a tensor with masked data replaced by default fill value,
// or by optional passed value
func (t *Dense) Filled(val ...interface{}) (interface{}, error) {
	tc := t.Clone().(*Dense)
	if !t.IsMasked() {
		return tc, nil
	}
	fillval := t.FillValue()
	if len(val) > 0 {
		fillval = val[0]
	}
	switch {
	case tc.IsScalar():
		if tc.mask[0] {
			tc.Set(0, fillval)
		}
	case tc.IsRowVec() || tc.IsColVec():
		sliceList := t.FlatMaskedContiguous()

		for i := range sliceList {
			tt, err := tc.Slice(nil, sliceList[i])
			if err != nil {
				ts := tt.(*Dense)
				ts.Memset(fillval)
			}
		}
	default:
		it := IteratorFromDense(tc)
		for i, _, err := it.NextInvalid(); err == nil; i, _, err = it.NextInvalid() {
			tc.Set(i, fillval)
		}
	}

	return tc, nil
}

// FilledInplace replaces masked data with default fill value,
// or by optional passed value
func (t *Dense) FilledInplace(val ...interface{}) (interface{}, error) {
	if !t.IsMasked() {
		return t, nil
	}
	fillval := t.FillValue()
	if len(val) > 0 {
		fillval = val[0]
	}
	switch {
	case t.IsScalar():
		if t.mask[0] {
			t.Set(0, fillval)
		}
	case t.IsRowVec() || t.IsColVec():
		sliceList := t.FlatMaskedContiguous()

		for i := range sliceList {
			tt, err := t.Slice(nil, sliceList[i])
			if err != nil {
				ts := tt.(*Dense)
				ts.Memset(fillval)
			}
		}
	default:
		it := IteratorFromDense(t)
		for i, _, err := it.NextInvalid(); err == nil; i, _, err = it.NextInvalid() {
			t.Set(i, fillval)
		}
	}

	return t, nil
}
