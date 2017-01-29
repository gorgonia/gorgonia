package tensor

import (
	"reflect"
	"unsafe"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func reduceRef(f reflect.Value, fnT reflect.Type, def reflect.Value, l *Dense) interface{} {
	retVal := def
	if l.len() == 0 {
		return retVal.Interface()
	}

	args := make([]reflect.Value, 0, fnT.NumIn())
	for i := 0; i < l.len(); i++ {
		v := reflect.ValueOf(l.get(i))
		args = append(args, retVal)
		args = append(args, v)
		retVal = f.Call(args)[0]
		args = args[:0]
	}
	return retVal.Interface()
}

func reduceB(f func(a, b bool) bool, def bool, l ...bool) (retVal bool) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceI(f func(a, b int) int, def int, l ...int) (retVal int) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceI8(f func(a, b int8) int8, def int8, l ...int8) (retVal int8) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceI16(f func(a, b int16) int16, def int16, l ...int16) (retVal int16) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceI32(f func(a, b int32) int32, def int32, l ...int32) (retVal int32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceI64(f func(a, b int64) int64, def int64, l ...int64) (retVal int64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceU(f func(a, b uint) uint, def uint, l ...uint) (retVal uint) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceU8(f func(a, b uint8) uint8, def uint8, l ...uint8) (retVal uint8) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceU16(f func(a, b uint16) uint16, def uint16, l ...uint16) (retVal uint16) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceU32(f func(a, b uint32) uint32, def uint32, l ...uint32) (retVal uint32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceU64(f func(a, b uint64) uint64, def uint64, l ...uint64) (retVal uint64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceUintptr(f func(a, b uintptr) uintptr, def uintptr, l ...uintptr) (retVal uintptr) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceF32(f func(a, b float32) float32, def float32, l ...float32) (retVal float32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceF64(f func(a, b float64) float64, def float64, l ...float64) (retVal float64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceC64(f func(a, b complex64) complex64, def complex64, l ...complex64) (retVal complex64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceC128(f func(a, b complex128) complex128, def complex128, l ...complex128) (retVal complex128) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceStr(f func(a, b string) string, def string, l ...string) (retVal string) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func reduceUnsafePointer(f func(a, b unsafe.Pointer) unsafe.Pointer, def unsafe.Pointer, l ...unsafe.Pointer) (retVal unsafe.Pointer) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}
