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

func sumI(a []int) int                  { return reduceI(addI, 0, a...) }
func sumI8(a []int8) int8               { return reduceI8(addI8, 0, a...) }
func sumI16(a []int16) int16            { return reduceI16(addI16, 0, a...) }
func sumI32(a []int32) int32            { return reduceI32(addI32, 0, a...) }
func sumI64(a []int64) int64            { return reduceI64(addI64, 0, a...) }
func sumU(a []uint) uint                { return reduceU(addU, 0, a...) }
func sumU8(a []uint8) uint8             { return reduceU8(addU8, 0, a...) }
func sumU16(a []uint16) uint16          { return reduceU16(addU16, 0, a...) }
func sumU32(a []uint32) uint32          { return reduceU32(addU32, 0, a...) }
func sumU64(a []uint64) uint64          { return reduceU64(addU64, 0, a...) }
func sumF32(a []float32) float32        { return reduceF32(addF32, 0, a...) }
func sumF64(a []float64) float64        { return reduceF64(addF64, 0, a...) }
func sumC64(a []complex64) complex64    { return reduceC64(addC64, 0, a...) }
func sumC128(a []complex128) complex128 { return reduceC128(addC128, 0, a...) }
func sliceMinI(a []int) int {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI(minI, a[0], a[1:]...)
}

func sliceMaxI(a []int) int {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI(maxI, a[0], a[1:]...)
}

func sliceMinI8(a []int8) int8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI8(minI8, a[0], a[1:]...)
}

func sliceMaxI8(a []int8) int8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI8(maxI8, a[0], a[1:]...)
}

func sliceMinI16(a []int16) int16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI16(minI16, a[0], a[1:]...)
}

func sliceMaxI16(a []int16) int16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI16(maxI16, a[0], a[1:]...)
}

func sliceMinI32(a []int32) int32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI32(minI32, a[0], a[1:]...)
}

func sliceMaxI32(a []int32) int32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI32(maxI32, a[0], a[1:]...)
}

func sliceMinI64(a []int64) int64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI64(minI64, a[0], a[1:]...)
}

func sliceMaxI64(a []int64) int64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceI64(maxI64, a[0], a[1:]...)
}

func sliceMinU(a []uint) uint {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU(minU, a[0], a[1:]...)
}

func sliceMaxU(a []uint) uint {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU(maxU, a[0], a[1:]...)
}

func sliceMinU8(a []uint8) uint8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU8(minU8, a[0], a[1:]...)
}

func sliceMaxU8(a []uint8) uint8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU8(maxU8, a[0], a[1:]...)
}

func sliceMinU16(a []uint16) uint16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU16(minU16, a[0], a[1:]...)
}

func sliceMaxU16(a []uint16) uint16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU16(maxU16, a[0], a[1:]...)
}

func sliceMinU32(a []uint32) uint32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU32(minU32, a[0], a[1:]...)
}

func sliceMaxU32(a []uint32) uint32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU32(maxU32, a[0], a[1:]...)
}

func sliceMinU64(a []uint64) uint64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU64(minU64, a[0], a[1:]...)
}

func sliceMaxU64(a []uint64) uint64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceU64(maxU64, a[0], a[1:]...)
}

func sliceMinF32(a []float32) float32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceF32(minF32, a[0], a[1:]...)
}

func sliceMaxF32(a []float32) float32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceF32(maxF32, a[0], a[1:]...)
}

func sliceMinF64(a []float64) float64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceF64(minF64, a[0], a[1:]...)
}

func sliceMaxF64(a []float64) float64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduceF64(maxF64, a[0], a[1:]...)
}
