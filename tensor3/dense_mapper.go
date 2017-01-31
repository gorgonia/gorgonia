package tensor

import (
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func (t *Dense) mapFn(fn interface{}, incr bool) (err error) {
	switch t.t.Kind() {
	case reflect.Bool:
		if f, ok := fn.(func(bool) bool); ok {
			data := t.bools()
			for i, v := range data {
				data[i] = f(v)
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(bool) bool", fn)
	case reflect.Int:
		if f, ok := fn.(func(int) int); ok {
			data := t.ints()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int) int", fn)
	case reflect.Int8:
		if f, ok := fn.(func(int8) int8); ok {
			data := t.int8s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int8) int8", fn)
	case reflect.Int16:
		if f, ok := fn.(func(int16) int16); ok {
			data := t.int16s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int16) int16", fn)
	case reflect.Int32:
		if f, ok := fn.(func(int32) int32); ok {
			data := t.int32s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int32) int32", fn)
	case reflect.Int64:
		if f, ok := fn.(func(int64) int64); ok {
			data := t.int64s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int64) int64", fn)
	case reflect.Uint:
		if f, ok := fn.(func(uint) uint); ok {
			data := t.uints()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint) uint", fn)
	case reflect.Uint8:
		if f, ok := fn.(func(uint8) uint8); ok {
			data := t.uint8s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint8) uint8", fn)
	case reflect.Uint16:
		if f, ok := fn.(func(uint16) uint16); ok {
			data := t.uint16s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint16) uint16", fn)
	case reflect.Uint32:
		if f, ok := fn.(func(uint32) uint32); ok {
			data := t.uint32s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint32) uint32", fn)
	case reflect.Uint64:
		if f, ok := fn.(func(uint64) uint64); ok {
			data := t.uint64s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint64) uint64", fn)
	case reflect.Uintptr:
		if f, ok := fn.(func(uintptr) uintptr); ok {
			data := t.uintptrs()
			for i, v := range data {
				data[i] = f(v)
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uintptr) uintptr", fn)
	case reflect.Float32:
		if f, ok := fn.(func(float32) float32); ok {
			data := t.float32s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(float32) float32", fn)
	case reflect.Float64:
		if f, ok := fn.(func(float64) float64); ok {
			data := t.float64s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(float64) float64", fn)
	case reflect.Complex64:
		if f, ok := fn.(func(complex64) complex64); ok {
			data := t.complex64s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(complex64) complex64", fn)
	case reflect.Complex128:
		if f, ok := fn.(func(complex128) complex128); ok {
			data := t.complex128s()
			for i, v := range data {
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(complex128) complex128", fn)
	case reflect.String:
		if f, ok := fn.(func(string) string); ok {
			data := t.strings()
			for i, v := range data {
				data[i] = f(v)
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(string) string", fn)
	case reflect.UnsafePointer:
		if f, ok := fn.(func(unsafe.Pointer) unsafe.Pointer); ok {
			data := t.unsafePointers()
			for i, v := range data {
				data[i] = f(v)
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(unsafe.Pointer) unsafe.Pointer", fn)
	default:
		// TODO: fix to handle incr
		var f reflect.Value
		var fnT reflect.Type
		if f, fnT, err = reductionFnType(fn, t.t.Type); err != nil {
			return
		}
		args := make([]reflect.Value, 0, fnT.NumIn())
		for i := 0; i < t.len(); i++ {
			args = append(args, reflect.ValueOf(t.get(i)))
			t.set(i, f.Call(args)[0].Interface())
			args = args[:0]
		}
	}
	return nil
}

func (t *Dense) iterMap(fn interface{}, it *FlatIterator, incr bool) (err error) {
	switch t.t.Kind() {
	case reflect.Bool:
		if f, ok := fn.(func(bool) bool); ok {
			data := t.bools()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				data[i] = f(v)
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(bool) bool", fn)
	case reflect.Int:
		if f, ok := fn.(func(int) int); ok {
			data := t.ints()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int) int", fn)
	case reflect.Int8:
		if f, ok := fn.(func(int8) int8); ok {
			data := t.int8s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int8) int8", fn)
	case reflect.Int16:
		if f, ok := fn.(func(int16) int16); ok {
			data := t.int16s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int16) int16", fn)
	case reflect.Int32:
		if f, ok := fn.(func(int32) int32); ok {
			data := t.int32s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int32) int32", fn)
	case reflect.Int64:
		if f, ok := fn.(func(int64) int64); ok {
			data := t.int64s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(int64) int64", fn)
	case reflect.Uint:
		if f, ok := fn.(func(uint) uint); ok {
			data := t.uints()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint) uint", fn)
	case reflect.Uint8:
		if f, ok := fn.(func(uint8) uint8); ok {
			data := t.uint8s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint8) uint8", fn)
	case reflect.Uint16:
		if f, ok := fn.(func(uint16) uint16); ok {
			data := t.uint16s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint16) uint16", fn)
	case reflect.Uint32:
		if f, ok := fn.(func(uint32) uint32); ok {
			data := t.uint32s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint32) uint32", fn)
	case reflect.Uint64:
		if f, ok := fn.(func(uint64) uint64); ok {
			data := t.uint64s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uint64) uint64", fn)
	case reflect.Uintptr:
		if f, ok := fn.(func(uintptr) uintptr); ok {
			data := t.uintptrs()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				data[i] = f(v)
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(uintptr) uintptr", fn)
	case reflect.Float32:
		if f, ok := fn.(func(float32) float32); ok {
			data := t.float32s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(float32) float32", fn)
	case reflect.Float64:
		if f, ok := fn.(func(float64) float64); ok {
			data := t.float64s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(float64) float64", fn)
	case reflect.Complex64:
		if f, ok := fn.(func(complex64) complex64); ok {
			data := t.complex64s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(complex64) complex64", fn)
	case reflect.Complex128:
		if f, ok := fn.(func(complex128) complex128); ok {
			data := t.complex128s()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				if incr {
					data[i] += f(v)
				} else {
					data[i] = f(v)
				}
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(complex128) complex128", fn)
	case reflect.String:
		if f, ok := fn.(func(string) string); ok {
			data := t.strings()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				data[i] = f(v)
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(string) string", fn)
	case reflect.UnsafePointer:
		if f, ok := fn.(func(unsafe.Pointer) unsafe.Pointer); ok {
			data := t.unsafePointers()
			var i int
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				v := data[i]
				data[i] = f(v)
			}
			if _, noop := err.(NoOpError); !noop {
				return
			}
			return nil
		}
		return errors.Errorf(extractionFail, "func(unsafe.Pointer) unsafe.Pointer", fn)
	default:
		// TODO: fix to handle incr
		var f reflect.Value
		var fnT reflect.Type
		if f, fnT, err = reductionFnType(fn, t.t.Type); err != nil {
			return
		}
		args := make([]reflect.Value, 0, fnT.NumIn())
		var i int
		for i, err = it.Next(); err == nil; i, err = it.Next() {
			args = append(args, reflect.ValueOf(t.get(i)))
			t.set(i, f.Call(args)[0].Interface())
			args = args[:0]
		}
		if _, noop := err.(NoOpError); !noop {
			return
		}
	}
	return nil
}
