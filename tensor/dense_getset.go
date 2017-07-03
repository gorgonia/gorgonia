package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func (t *Dense) memsetIter(x interface{}) (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.UnsafePointers()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.ptr)
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
		err = handleNoOp(err)
	}
	return
}

func (t *Dense) zeroIter() (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.Bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = false

		}
		err = handleNoOp(err)
	case reflect.Int:
		data := t.Ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int8:
		data := t.Int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int16:
		data := t.Int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int32:
		data := t.Int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int64:
		data := t.Int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint:
		data := t.Uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		data := t.Uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		data := t.Uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		data := t.Uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		data := t.Uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		data := t.Uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float32:
		data := t.Float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float64:
		data := t.Float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		data := t.Complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		data := t.Complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.String:
		data := t.Strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = ""

		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		data := t.UnsafePointers()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = nil

		}
		err = handleNoOp(err)
	default:
		ptr := uintptr(t.ptr)
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(t.t))
		}
		err = handleNoOp(err)
	}
	return
}

func copySliced(dest *Dense, dstart, dend int, src *Dense, sstart, send int) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if src.IsMasked() {
		mask := dest.mask
		if cap(dest.mask) < dend {
			mask = make([]bool, dend)
		}
		copy(mask, dest.mask)
		dest.mask = mask
		copy(dest.mask[dstart:dend], src.mask[sstart:send])
	}
	switch dest.t.Kind() {
	case reflect.Bool:
		return copy(dest.Bools()[dstart:dend], src.Bools()[sstart:send])
	case reflect.Int:
		return copy(dest.Ints()[dstart:dend], src.Ints()[sstart:send])
	case reflect.Int8:
		return copy(dest.Int8s()[dstart:dend], src.Int8s()[sstart:send])
	case reflect.Int16:
		return copy(dest.Int16s()[dstart:dend], src.Int16s()[sstart:send])
	case reflect.Int32:
		return copy(dest.Int32s()[dstart:dend], src.Int32s()[sstart:send])
	case reflect.Int64:
		return copy(dest.Int64s()[dstart:dend], src.Int64s()[sstart:send])
	case reflect.Uint:
		return copy(dest.Uints()[dstart:dend], src.Uints()[sstart:send])
	case reflect.Uint8:
		return copy(dest.Uint8s()[dstart:dend], src.Uint8s()[sstart:send])
	case reflect.Uint16:
		return copy(dest.Uint16s()[dstart:dend], src.Uint16s()[sstart:send])
	case reflect.Uint32:
		return copy(dest.Uint32s()[dstart:dend], src.Uint32s()[sstart:send])
	case reflect.Uint64:
		return copy(dest.Uint64s()[dstart:dend], src.Uint64s()[sstart:send])
	case reflect.Uintptr:
		return copy(dest.Uintptrs()[dstart:dend], src.Uintptrs()[sstart:send])
	case reflect.Float32:
		return copy(dest.Float32s()[dstart:dend], src.Float32s()[sstart:send])
	case reflect.Float64:
		return copy(dest.Float64s()[dstart:dend], src.Float64s()[sstart:send])
	case reflect.Complex64:
		return copy(dest.Complex64s()[dstart:dend], src.Complex64s()[sstart:send])
	case reflect.Complex128:
		return copy(dest.Complex128s()[dstart:dend], src.Complex128s()[sstart:send])

	case reflect.String:
		return copy(dest.Strings()[dstart:dend], src.Strings()[sstart:send])

	case reflect.UnsafePointer:
		return copy(dest.UnsafePointers()[dstart:dend], src.UnsafePointers()[sstart:send])
	default:
		dv := reflect.ValueOf(dest.v)
		dv = dv.Slice(dstart, dend)
		sv := reflect.ValueOf(src.v)
		sv = sv.Slice(sstart, send)
		return reflect.Copy(dv, sv)
	}
}

func copyDenseIter(dest, src *Dense, diter, siter *FlatIterator) (int, error) {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil && !dest.IsMaterializable() && !src.IsMaterializable() {
		return copyDense(dest, src), nil
	}

	if diter == nil {
		diter = NewFlatIterator(dest.AP)
	}
	if siter == nil {
		siter = NewFlatIterator(src.AP)
	}

	isMasked := src.IsMasked()
	if isMasked {
		if cap(dest.mask) < src.DataSize() {
			dest.mask = make([]bool, src.DataSize())
		}
		dest.mask = dest.mask[:dest.DataSize()]
	}

	k := dest.t.Kind()
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		if j, err = siter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		if isMasked {
			dest.mask[i] = src.mask[j]
		}

		switch k {
		case reflect.Bool:
			dest.SetB(i, src.GetB(j))
		case reflect.Int:
			dest.SetI(i, src.GetI(j))
		case reflect.Int8:
			dest.SetI8(i, src.GetI8(j))
		case reflect.Int16:
			dest.SetI16(i, src.GetI16(j))
		case reflect.Int32:
			dest.SetI32(i, src.GetI32(j))
		case reflect.Int64:
			dest.SetI64(i, src.GetI64(j))
		case reflect.Uint:
			dest.SetU(i, src.GetU(j))
		case reflect.Uint8:
			dest.SetU8(i, src.GetU8(j))
		case reflect.Uint16:
			dest.SetU16(i, src.GetU16(j))
		case reflect.Uint32:
			dest.SetU32(i, src.GetU32(j))
		case reflect.Uint64:
			dest.SetU64(i, src.GetU64(j))
		case reflect.Uintptr:
			dest.SetUintptr(i, src.GetUintptr(j))
		case reflect.Float32:
			dest.SetF32(i, src.GetF32(j))
		case reflect.Float64:
			dest.SetF64(i, src.GetF64(j))
		case reflect.Complex64:
			dest.SetC64(i, src.GetC64(j))
		case reflect.Complex128:
			dest.SetC128(i, src.GetC128(j))
		case reflect.String:
			dest.SetStr(i, src.GetStr(j))
		case reflect.UnsafePointer:
			dest.SetUnsafePointer(i, src.GetUnsafePointer(j))
		default:
			dest.Set(i, src.Get(j))
		}
		count++
	}
	return count, err
}

// the method assumes the AP and metadata has already been set and this is simply slicing the values
func (t *Dense) slice(start, end int) {
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.Bools()[start:end]
		t.fromSlice(data)
	case reflect.Int:
		data := t.Ints()[start:end]
		t.fromSlice(data)
	case reflect.Int8:
		data := t.Int8s()[start:end]
		t.fromSlice(data)
	case reflect.Int16:
		data := t.Int16s()[start:end]
		t.fromSlice(data)
	case reflect.Int32:
		data := t.Int32s()[start:end]
		t.fromSlice(data)
	case reflect.Int64:
		data := t.Int64s()[start:end]
		t.fromSlice(data)
	case reflect.Uint:
		data := t.Uints()[start:end]
		t.fromSlice(data)
	case reflect.Uint8:
		data := t.Uint8s()[start:end]
		t.fromSlice(data)
	case reflect.Uint16:
		data := t.Uint16s()[start:end]
		t.fromSlice(data)
	case reflect.Uint32:
		data := t.Uint32s()[start:end]
		t.fromSlice(data)
	case reflect.Uint64:
		data := t.Uint64s()[start:end]
		t.fromSlice(data)
	case reflect.Uintptr:
		data := t.Uintptrs()[start:end]
		t.fromSlice(data)
	case reflect.Float32:
		data := t.Float32s()[start:end]
		t.fromSlice(data)
	case reflect.Float64:
		data := t.Float64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex64:
		data := t.Complex64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex128:
		data := t.Complex128s()[start:end]
		t.fromSlice(data)

	case reflect.String:
		data := t.Strings()[start:end]
		t.fromSlice(data)

	case reflect.UnsafePointer:
		data := t.UnsafePointers()[start:end]
		t.fromSlice(data)
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}
}
