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
	switch t.t {
	case Bool:
		xv, ok := x.(bool)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Int:
		xv, ok := x.(int)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Int8:
		xv, ok := x.(int8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Int16:
		xv, ok := x.(int16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Int32:
		xv, ok := x.(int32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Int64:
		xv, ok := x.(int64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uint:
		xv, ok := x.(uint)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uint8:
		xv, ok := x.(uint8)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uint16:
		xv, ok := x.(uint16)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uint32:
		xv, ok := x.(uint32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uint64:
		xv, ok := x.(uint64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Uintptr:
		xv, ok := x.(uintptr)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Float32:
		xv, ok := x.(float32)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Float64:
		xv, ok := x.(float64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Complex64:
		xv, ok := x.(complex64)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case Complex128:
		xv, ok := x.(complex128)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case String:
		xv, ok := x.(string)
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.Strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = xv
		}
		err = handleNoOp(err)
	case UnsafePointer:
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
		ptr := uintptr(t.array.Ptr)
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
	switch t.t {
	case Bool:
		data := t.Bools()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = false

		}
		err = handleNoOp(err)
	case Int:
		data := t.Ints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Int8:
		data := t.Int8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Int16:
		data := t.Int16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Int32:
		data := t.Int32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Int64:
		data := t.Int64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uint:
		data := t.Uints()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uint8:
		data := t.Uint8s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uint16:
		data := t.Uint16s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uint32:
		data := t.Uint32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uint64:
		data := t.Uint64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Uintptr:
		data := t.Uintptrs()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Float32:
		data := t.Float32s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Float64:
		data := t.Float64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Complex64:
		data := t.Complex64s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case Complex128:
		data := t.Complex128s()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = 0
		}
		err = handleNoOp(err)
	case String:
		data := t.Strings()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = ""

		}
		err = handleNoOp(err)
	case UnsafePointer:
		data := t.UnsafePointers()
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			data[i] = nil

		}
		err = handleNoOp(err)
	default:
		ptr := uintptr(t.array.Ptr)
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

// the method assumes the AP and metadata has already been set and this is simply slicing the values
func (t *Dense) slice(start, end int) {
	if !t.IsNativelyAccessible() {
		t.sliceInaccessible(start, end)
		return
	}

	switch t.t.Size() {
	case 0:
	case 1:
	case 2:
	case 4:
	case 8:
	default:
	}

	switch t.t {
	case Bool:
		data := t.Bools()[start:end]
		t.fromSlice(data)
	case Int:
		data := t.Ints()[start:end]
		t.fromSlice(data)
	case Int8:
		data := t.Int8s()[start:end]
		t.fromSlice(data)
	case Int16:
		data := t.Int16s()[start:end]
		t.fromSlice(data)
	case Int32:
		data := t.Int32s()[start:end]
		t.fromSlice(data)
	case Int64:
		data := t.Int64s()[start:end]
		t.fromSlice(data)
	case Uint:
		data := t.Uints()[start:end]
		t.fromSlice(data)
	case Uint8:
		data := t.Uint8s()[start:end]
		t.fromSlice(data)
	case Uint16:
		data := t.Uint16s()[start:end]
		t.fromSlice(data)
	case Uint32:
		data := t.Uint32s()[start:end]
		t.fromSlice(data)
	case Uint64:
		data := t.Uint64s()[start:end]
		t.fromSlice(data)
	case Uintptr:
		data := t.Uintptrs()[start:end]
		t.fromSlice(data)
	case Float32:
		data := t.Float32s()[start:end]
		t.fromSlice(data)
	case Float64:
		data := t.Float64s()[start:end]
		t.fromSlice(data)
	case Complex64:
		data := t.Complex64s()[start:end]
		t.fromSlice(data)
	case Complex128:
		data := t.Complex128s()[start:end]
		t.fromSlice(data)

	case String:
		data := t.Strings()[start:end]
		t.fromSlice(data)

	case UnsafePointer:
		data := t.UnsafePointers()[start:end]
		t.fromSlice(data)
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}
}

func (t *Dense) sliceInaccessible(start, end int) {

}
