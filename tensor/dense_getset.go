package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

// Memset sets all values in the *Dense tensor to x.
func (t *Dense) Memset(x interface{}) error {
	if t.IsMaterializable() {
		return t.memsetIter(x)
	}
	switch t.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.bools()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.ints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uints()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint8s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint16s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uintptrs()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float32s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex64s()
		for i := range data {
			data[i] = xv
		}
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex128s()
		for i := range data {
			data[i] = xv
		}
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.strings()
		for i := range data {
			data[i] = xv
		}
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.unsafePointers()
		for i := range data {
			data[i] = xv
		}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.ptr)
		for i := 0; i < t.l; i++ {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
	}
	return nil
}

func (t *Dense) memsetIter(x interface{}) (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		xv, ok := x.(unsafe.Pointer)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.unsafePointers()
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

// Zero zeroes out the underlying array of the *Dense tensor
func (t *Dense) Zero() {
	if t.IsMaterializable() {
		if err := t.zeroIter(); err != nil {
			panic(err)
		}
	}
	if t.IsMasked() {
		t.ResetMask()
	}
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		for i := range data {
			data[i] = false

		}
	case reflect.Int:
		data := t.ints()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int8:
		data := t.int8s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int16:
		data := t.int16s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int32:
		data := t.int32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Int64:
		data := t.int64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint:
		data := t.uints()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint8:
		data := t.uint8s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint16:
		data := t.uint16s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint32:
		data := t.uint32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uint64:
		data := t.uint64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Uintptr:
		data := t.uintptrs()
		for i := range data {
			data[i] = 0
		}
	case reflect.Float32:
		data := t.float32s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Float64:
		data := t.float64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Complex64:
		data := t.complex64s()
		for i := range data {
			data[i] = 0
		}
	case reflect.Complex128:
		data := t.complex128s()
		for i := range data {
			data[i] = 0
		}
	case reflect.String:
		data := t.strings()
		for i := range data {
			data[i] = ""

		}
	case reflect.UnsafePointer:
		data := t.unsafePointers()
		for i := range data {
			data[i] = nil

		}
	default:
		ptr := uintptr(t.ptr)
		for i := 0; i < t.l; i++ {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(t.t))
		}
	}
}

func (t *Dense) zeroIter() (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	case reflect.Bool:
		data := t.bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = false

		}
		err = handleNoOp(err)
	case reflect.Int:
		data := t.ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int8:
		data := t.int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int16:
		data := t.int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int32:
		data := t.int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Int64:
		data := t.int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint:
		data := t.uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint8:
		data := t.uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint16:
		data := t.uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint32:
		data := t.uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uint64:
		data := t.uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Uintptr:
		data := t.uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float32:
		data := t.float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Float64:
		data := t.float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex64:
		data := t.complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.Complex128:
		data := t.complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case reflect.String:
		data := t.strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = ""

		}
		err = handleNoOp(err)
	case reflect.UnsafePointer:
		data := t.unsafePointers()
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

func copyDense(dest, src *Dense) int {
	if dest.t != src.t {
		err := errors.Errorf(dtypeMismatch, src.t, dest.t)
		panic(err.Error())
	}
	if src.IsMasked() {
		if cap(dest.mask) < len(src.mask) {
			dest.mask = make([]bool, len(src.mask))
		}
		copy(dest.mask, src.mask)
		dest.mask = dest.mask[:len(src.mask)]
	}
	switch dest.t.Kind() {
	case reflect.Bool:
		return copy(dest.bools(), src.bools())
	case reflect.Int:
		return copy(dest.ints(), src.ints())
	case reflect.Int8:
		return copy(dest.int8s(), src.int8s())
	case reflect.Int16:
		return copy(dest.int16s(), src.int16s())
	case reflect.Int32:
		return copy(dest.int32s(), src.int32s())
	case reflect.Int64:
		return copy(dest.int64s(), src.int64s())
	case reflect.Uint:
		return copy(dest.uints(), src.uints())
	case reflect.Uint8:
		return copy(dest.uint8s(), src.uint8s())
	case reflect.Uint16:
		return copy(dest.uint16s(), src.uint16s())
	case reflect.Uint32:
		return copy(dest.uint32s(), src.uint32s())
	case reflect.Uint64:
		return copy(dest.uint64s(), src.uint64s())
	case reflect.Uintptr:
		return copy(dest.uintptrs(), src.uintptrs())
	case reflect.Float32:
		return copy(dest.float32s(), src.float32s())
	case reflect.Float64:
		return copy(dest.float64s(), src.float64s())
	case reflect.Complex64:
		return copy(dest.complex64s(), src.complex64s())
	case reflect.Complex128:
		return copy(dest.complex128s(), src.complex128s())

	case reflect.String:
		return copy(dest.strings(), src.strings())

	case reflect.UnsafePointer:
		return copy(dest.unsafePointers(), src.unsafePointers())
	default:
		dv := reflect.ValueOf(dest.v)
		sv := reflect.ValueOf(src.v)
		return reflect.Copy(dv, sv)
	}
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
		return copy(dest.bools()[dstart:dend], src.bools()[sstart:send])
	case reflect.Int:
		return copy(dest.ints()[dstart:dend], src.ints()[sstart:send])
	case reflect.Int8:
		return copy(dest.int8s()[dstart:dend], src.int8s()[sstart:send])
	case reflect.Int16:
		return copy(dest.int16s()[dstart:dend], src.int16s()[sstart:send])
	case reflect.Int32:
		return copy(dest.int32s()[dstart:dend], src.int32s()[sstart:send])
	case reflect.Int64:
		return copy(dest.int64s()[dstart:dend], src.int64s()[sstart:send])
	case reflect.Uint:
		return copy(dest.uints()[dstart:dend], src.uints()[sstart:send])
	case reflect.Uint8:
		return copy(dest.uint8s()[dstart:dend], src.uint8s()[sstart:send])
	case reflect.Uint16:
		return copy(dest.uint16s()[dstart:dend], src.uint16s()[sstart:send])
	case reflect.Uint32:
		return copy(dest.uint32s()[dstart:dend], src.uint32s()[sstart:send])
	case reflect.Uint64:
		return copy(dest.uint64s()[dstart:dend], src.uint64s()[sstart:send])
	case reflect.Uintptr:
		return copy(dest.uintptrs()[dstart:dend], src.uintptrs()[sstart:send])
	case reflect.Float32:
		return copy(dest.float32s()[dstart:dend], src.float32s()[sstart:send])
	case reflect.Float64:
		return copy(dest.float64s()[dstart:dend], src.float64s()[sstart:send])
	case reflect.Complex64:
		return copy(dest.complex64s()[dstart:dend], src.complex64s()[sstart:send])
	case reflect.Complex128:
		return copy(dest.complex128s()[dstart:dend], src.complex128s()[sstart:send])

	case reflect.String:
		return copy(dest.strings()[dstart:dend], src.strings()[sstart:send])

	case reflect.UnsafePointer:
		return copy(dest.unsafePointers()[dstart:dend], src.unsafePointers()[sstart:send])
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
			dest.setB(i, src.getB(j))
		case reflect.Int:
			dest.setI(i, src.getI(j))
		case reflect.Int8:
			dest.setI8(i, src.getI8(j))
		case reflect.Int16:
			dest.setI16(i, src.getI16(j))
		case reflect.Int32:
			dest.setI32(i, src.getI32(j))
		case reflect.Int64:
			dest.setI64(i, src.getI64(j))
		case reflect.Uint:
			dest.setU(i, src.getU(j))
		case reflect.Uint8:
			dest.setU8(i, src.getU8(j))
		case reflect.Uint16:
			dest.setU16(i, src.getU16(j))
		case reflect.Uint32:
			dest.setU32(i, src.getU32(j))
		case reflect.Uint64:
			dest.setU64(i, src.getU64(j))
		case reflect.Uintptr:
			dest.setUintptr(i, src.getUintptr(j))
		case reflect.Float32:
			dest.setF32(i, src.getF32(j))
		case reflect.Float64:
			dest.setF64(i, src.getF64(j))
		case reflect.Complex64:
			dest.setC64(i, src.getC64(j))
		case reflect.Complex128:
			dest.setC128(i, src.getC128(j))
		case reflect.String:
			dest.setStr(i, src.getStr(j))
		case reflect.UnsafePointer:
			dest.setUnsafePointer(i, src.getUnsafePointer(j))
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
		data := t.bools()[start:end]
		t.fromSlice(data)
	case reflect.Int:
		data := t.ints()[start:end]
		t.fromSlice(data)
	case reflect.Int8:
		data := t.int8s()[start:end]
		t.fromSlice(data)
	case reflect.Int16:
		data := t.int16s()[start:end]
		t.fromSlice(data)
	case reflect.Int32:
		data := t.int32s()[start:end]
		t.fromSlice(data)
	case reflect.Int64:
		data := t.int64s()[start:end]
		t.fromSlice(data)
	case reflect.Uint:
		data := t.uints()[start:end]
		t.fromSlice(data)
	case reflect.Uint8:
		data := t.uint8s()[start:end]
		t.fromSlice(data)
	case reflect.Uint16:
		data := t.uint16s()[start:end]
		t.fromSlice(data)
	case reflect.Uint32:
		data := t.uint32s()[start:end]
		t.fromSlice(data)
	case reflect.Uint64:
		data := t.uint64s()[start:end]
		t.fromSlice(data)
	case reflect.Uintptr:
		data := t.uintptrs()[start:end]
		t.fromSlice(data)
	case reflect.Float32:
		data := t.float32s()[start:end]
		t.fromSlice(data)
	case reflect.Float64:
		data := t.float64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex64:
		data := t.complex64s()[start:end]
		t.fromSlice(data)
	case reflect.Complex128:
		data := t.complex128s()[start:end]
		t.fromSlice(data)

	case reflect.String:
		data := t.strings()[start:end]
		t.fromSlice(data)

	case reflect.UnsafePointer:
		data := t.unsafePointers()[start:end]
		t.fromSlice(data)
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}
}

// Eq checks that any two things are equal. If the shapes are the same, but the strides are not the same, it's will still be considered the same
func (t *Dense) Eq(other interface{}) bool {
	if ot, ok := other.(*Dense); ok {
		if ot == t {
			return true
		}

		if ot.len() != t.len() {
			return false
		}

		if t.t != ot.t {
			return false
		}

		if !t.Shape().Eq(ot.Shape()) {
			return false
		}

		switch t.t.Kind() {
		case reflect.Bool:
			for i, v := range t.bools() {
				if ot.getB(i) != v {
					return false
				}
			}
		case reflect.Int:
			for i, v := range t.ints() {
				if ot.getI(i) != v {
					return false
				}
			}
		case reflect.Int8:
			for i, v := range t.int8s() {
				if ot.getI8(i) != v {
					return false
				}
			}
		case reflect.Int16:
			for i, v := range t.int16s() {
				if ot.getI16(i) != v {
					return false
				}
			}
		case reflect.Int32:
			for i, v := range t.int32s() {
				if ot.getI32(i) != v {
					return false
				}
			}
		case reflect.Int64:
			for i, v := range t.int64s() {
				if ot.getI64(i) != v {
					return false
				}
			}
		case reflect.Uint:
			for i, v := range t.uints() {
				if ot.getU(i) != v {
					return false
				}
			}
		case reflect.Uint8:
			for i, v := range t.uint8s() {
				if ot.getU8(i) != v {
					return false
				}
			}
		case reflect.Uint16:
			for i, v := range t.uint16s() {
				if ot.getU16(i) != v {
					return false
				}
			}
		case reflect.Uint32:
			for i, v := range t.uint32s() {
				if ot.getU32(i) != v {
					return false
				}
			}
		case reflect.Uint64:
			for i, v := range t.uint64s() {
				if ot.getU64(i) != v {
					return false
				}
			}
		case reflect.Uintptr:
			for i, v := range t.uintptrs() {
				if ot.getUintptr(i) != v {
					return false
				}
			}
		case reflect.Float32:
			for i, v := range t.float32s() {
				if ot.getF32(i) != v {
					return false
				}
			}
		case reflect.Float64:
			for i, v := range t.float64s() {
				if ot.getF64(i) != v {
					return false
				}
			}
		case reflect.Complex64:
			for i, v := range t.complex64s() {
				if ot.getC64(i) != v {
					return false
				}
			}
		case reflect.Complex128:
			for i, v := range t.complex128s() {
				if ot.getC128(i) != v {
					return false
				}
			}
		case reflect.String:
			for i, v := range t.strings() {
				if ot.getStr(i) != v {
					return false
				}
			}
		case reflect.UnsafePointer:
			for i, v := range t.unsafePointers() {
				if ot.getUnsafePointer(i) != v {
					return false
				}
			}
		default:
			for i := 0; i < t.len(); i++ {
				if !reflect.DeepEqual(t.Get(i), ot.Get(i)) {
					return false
				}
			}
		}
		return true
	}
	return false
}
