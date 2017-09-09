package execution

import "unsafe"

/*
GENERATED FILE. DO NOT EDIT
*/

func ReduceB(f func(a, b bool) bool, def bool, l ...bool) (retVal bool) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceI(f func(a, b int) int, def int, l ...int) (retVal int) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceI8(f func(a, b int8) int8, def int8, l ...int8) (retVal int8) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceI16(f func(a, b int16) int16, def int16, l ...int16) (retVal int16) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceI32(f func(a, b int32) int32, def int32, l ...int32) (retVal int32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceI64(f func(a, b int64) int64, def int64, l ...int64) (retVal int64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceU(f func(a, b uint) uint, def uint, l ...uint) (retVal uint) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceU8(f func(a, b uint8) uint8, def uint8, l ...uint8) (retVal uint8) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceU16(f func(a, b uint16) uint16, def uint16, l ...uint16) (retVal uint16) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceU32(f func(a, b uint32) uint32, def uint32, l ...uint32) (retVal uint32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceU64(f func(a, b uint64) uint64, def uint64, l ...uint64) (retVal uint64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceUintptr(f func(a, b uintptr) uintptr, def uintptr, l ...uintptr) (retVal uintptr) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceF32(f func(a, b float32) float32, def float32, l ...float32) (retVal float32) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceF64(f func(a, b float64) float64, def float64, l ...float64) (retVal float64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceC64(f func(a, b complex64) complex64, def complex64, l ...complex64) (retVal complex64) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceC128(f func(a, b complex128) complex128, def complex128, l ...complex128) (retVal complex128) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceStr(f func(a, b string) string, def string, l ...string) (retVal string) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func ReduceUnsafePointer(f func(a, b unsafe.Pointer) unsafe.Pointer, def unsafe.Pointer, l ...unsafe.Pointer) (retVal unsafe.Pointer) {
	retVal = def
	if len(l) == 0 {
		return
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

func SumI(a []int) int                   { return ReduceI(AddI, 0, a...) }
func SumI8(a []int8) int8                { return ReduceI8(AddI8, 0, a...) }
func SumI16(a []int16) int16             { return ReduceI16(AddI16, 0, a...) }
func SumI32(a []int32) int32             { return ReduceI32(AddI32, 0, a...) }
func SumI64(a []int64) int64             { return ReduceI64(AddI64, 0, a...) }
func SumU(a []uint) uint                 { return ReduceU(AddU, 0, a...) }
func SumU8(a []uint8) uint8              { return ReduceU8(AddU8, 0, a...) }
func SumU16(a []uint16) uint16           { return ReduceU16(AddU16, 0, a...) }
func SumU32(a []uint32) uint32           { return ReduceU32(AddU32, 0, a...) }
func SumU64(a []uint64) uint64           { return ReduceU64(AddU64, 0, a...) }
func SumF32(a []float32) float32         { return ReduceF32(AddF32, 0, a...) }
func SumF64(a []float64) float64         { return ReduceF64(AddF64, 0, a...) }
func SumC64(a []complex64) complex64     { return ReduceC64(AddC64, 0, a...) }
func SumC128(a []complex128) complex128  { return ReduceC128(AddC128, 0, a...) }
func ProdI(a []int) int                  { return ReduceI(MulI, 1, a...) }
func ProdI8(a []int8) int8               { return ReduceI8(MulI8, 1, a...) }
func ProdI16(a []int16) int16            { return ReduceI16(MulI16, 1, a...) }
func ProdI32(a []int32) int32            { return ReduceI32(MulI32, 1, a...) }
func ProdI64(a []int64) int64            { return ReduceI64(MulI64, 1, a...) }
func ProdU(a []uint) uint                { return ReduceU(MulU, 1, a...) }
func ProdU8(a []uint8) uint8             { return ReduceU8(MulU8, 1, a...) }
func ProdU16(a []uint16) uint16          { return ReduceU16(MulU16, 1, a...) }
func ProdU32(a []uint32) uint32          { return ReduceU32(MulU32, 1, a...) }
func ProdU64(a []uint64) uint64          { return ReduceU64(MulU64, 1, a...) }
func ProdF32(a []float32) float32        { return ReduceF32(MulF32, 1, a...) }
func ProdF64(a []float64) float64        { return ReduceF64(MulF64, 1, a...) }
func ProdC64(a []complex64) complex64    { return ReduceC64(MulC64, 1, a...) }
func ProdC128(a []complex128) complex128 { return ReduceC128(MulC128, 1, a...) }

func SliceMinI(a []int) int {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI(MinI, a[0], a[1:]...)
}

func SliceMaxI(a []int) int {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI(MaxI, a[0], a[1:]...)
}

func SliceMinI8(a []int8) int8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI8(MinI8, a[0], a[1:]...)
}

func SliceMaxI8(a []int8) int8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI8(MaxI8, a[0], a[1:]...)
}

func SliceMinI16(a []int16) int16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI16(MinI16, a[0], a[1:]...)
}

func SliceMaxI16(a []int16) int16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI16(MaxI16, a[0], a[1:]...)
}

func SliceMinI32(a []int32) int32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI32(MinI32, a[0], a[1:]...)
}

func SliceMaxI32(a []int32) int32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI32(MaxI32, a[0], a[1:]...)
}

func SliceMinI64(a []int64) int64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI64(MinI64, a[0], a[1:]...)
}

func SliceMaxI64(a []int64) int64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceI64(MaxI64, a[0], a[1:]...)
}

func SliceMinU(a []uint) uint {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU(MinU, a[0], a[1:]...)
}

func SliceMaxU(a []uint) uint {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU(MaxU, a[0], a[1:]...)
}

func SliceMinU8(a []uint8) uint8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU8(MinU8, a[0], a[1:]...)
}

func SliceMaxU8(a []uint8) uint8 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU8(MaxU8, a[0], a[1:]...)
}

func SliceMinU16(a []uint16) uint16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU16(MinU16, a[0], a[1:]...)
}

func SliceMaxU16(a []uint16) uint16 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU16(MaxU16, a[0], a[1:]...)
}

func SliceMinU32(a []uint32) uint32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU32(MinU32, a[0], a[1:]...)
}

func SliceMaxU32(a []uint32) uint32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU32(MaxU32, a[0], a[1:]...)
}

func SliceMinU64(a []uint64) uint64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU64(MinU64, a[0], a[1:]...)
}

func SliceMaxU64(a []uint64) uint64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceU64(MaxU64, a[0], a[1:]...)
}

func SliceMinF32(a []float32) float32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceF32(MinF32, a[0], a[1:]...)
}

func SliceMaxF32(a []float32) float32 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceF32(MaxF32, a[0], a[1:]...)
}

func SliceMinF64(a []float64) float64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceF64(MinF64, a[0], a[1:]...)
}

func SliceMaxF64(a []float64) float64 {
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return ReduceF64(MaxF64, a[0], a[1:]...)
}

func reduceFirstB(data, retVal []bool, split, size int, fn func(a, b []bool)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstB(data, retVal []bool, split, size int, fn func(a, b bool) bool) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstI(data, retVal []int, split, size int, fn func(a, b []int)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstI(data, retVal []int, split, size int, fn func(a, b int) int) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstI8(data, retVal []int8, split, size int, fn func(a, b []int8)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstI8(data, retVal []int8, split, size int, fn func(a, b int8) int8) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstI16(data, retVal []int16, split, size int, fn func(a, b []int16)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstI16(data, retVal []int16, split, size int, fn func(a, b int16) int16) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstI32(data, retVal []int32, split, size int, fn func(a, b []int32)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstI32(data, retVal []int32, split, size int, fn func(a, b int32) int32) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstI64(data, retVal []int64, split, size int, fn func(a, b []int64)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstI64(data, retVal []int64, split, size int, fn func(a, b int64) int64) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstU(data, retVal []uint, split, size int, fn func(a, b []uint)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstU(data, retVal []uint, split, size int, fn func(a, b uint) uint) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstU8(data, retVal []uint8, split, size int, fn func(a, b []uint8)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstU8(data, retVal []uint8, split, size int, fn func(a, b uint8) uint8) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstU16(data, retVal []uint16, split, size int, fn func(a, b []uint16)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstU16(data, retVal []uint16, split, size int, fn func(a, b uint16) uint16) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstU32(data, retVal []uint32, split, size int, fn func(a, b []uint32)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstU32(data, retVal []uint32, split, size int, fn func(a, b uint32) uint32) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstU64(data, retVal []uint64, split, size int, fn func(a, b []uint64)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstU64(data, retVal []uint64, split, size int, fn func(a, b uint64) uint64) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstUintptr(data, retVal []uintptr, split, size int, fn func(a, b []uintptr)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstUintptr(data, retVal []uintptr, split, size int, fn func(a, b uintptr) uintptr) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstF32(data, retVal []float32, split, size int, fn func(a, b []float32)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstF32(data, retVal []float32, split, size int, fn func(a, b float32) float32) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstF64(data, retVal []float64, split, size int, fn func(a, b []float64)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstF64(data, retVal []float64, split, size int, fn func(a, b float64) float64) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstC64(data, retVal []complex64, split, size int, fn func(a, b []complex64)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstC64(data, retVal []complex64, split, size int, fn func(a, b complex64) complex64) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstC128(data, retVal []complex128, split, size int, fn func(a, b []complex128)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstC128(data, retVal []complex128, split, size int, fn func(a, b complex128) complex128) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstStr(data, retVal []string, split, size int, fn func(a, b []string)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstStr(data, retVal []string, split, size int, fn func(a, b string) string) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceFirstUnsafePointer(data, retVal []unsafe.Pointer, split, size int, fn func(a, b []unsafe.Pointer)) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		fn(retVal, data[start:start+split])
		start += split
	}
}

func genericReduceFirstUnsafePointer(data, retVal []unsafe.Pointer, split, size int, fn func(a, b unsafe.Pointer) unsafe.Pointer) {
	start := split
	copy(retVal[0:split], data[0:split])
	for i := 0; i < size-1; i++ {
		for j := 0; j < split; j++ {
			retVal[j] = fn(retVal[j], data[j+start])
		}
		start += split
	}
}
func reduceLastB(a, retVal []bool, dimSize int, defaultValue bool, fn func(a []bool) bool) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastB(a, retVal []bool, dimSize int, defaultValue bool, fn func(bool, bool) bool) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceB(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastI(a, retVal []int, dimSize int, defaultValue int, fn func(a []int) int) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastI(a, retVal []int, dimSize int, defaultValue int, fn func(int, int) int) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceI(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastI8(a, retVal []int8, dimSize int, defaultValue int8, fn func(a []int8) int8) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastI8(a, retVal []int8, dimSize int, defaultValue int8, fn func(int8, int8) int8) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceI8(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastI16(a, retVal []int16, dimSize int, defaultValue int16, fn func(a []int16) int16) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastI16(a, retVal []int16, dimSize int, defaultValue int16, fn func(int16, int16) int16) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceI16(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastI32(a, retVal []int32, dimSize int, defaultValue int32, fn func(a []int32) int32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastI32(a, retVal []int32, dimSize int, defaultValue int32, fn func(int32, int32) int32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceI32(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastI64(a, retVal []int64, dimSize int, defaultValue int64, fn func(a []int64) int64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastI64(a, retVal []int64, dimSize int, defaultValue int64, fn func(int64, int64) int64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceI64(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastU(a, retVal []uint, dimSize int, defaultValue uint, fn func(a []uint) uint) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastU(a, retVal []uint, dimSize int, defaultValue uint, fn func(uint, uint) uint) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceU(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastU8(a, retVal []uint8, dimSize int, defaultValue uint8, fn func(a []uint8) uint8) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastU8(a, retVal []uint8, dimSize int, defaultValue uint8, fn func(uint8, uint8) uint8) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceU8(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastU16(a, retVal []uint16, dimSize int, defaultValue uint16, fn func(a []uint16) uint16) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastU16(a, retVal []uint16, dimSize int, defaultValue uint16, fn func(uint16, uint16) uint16) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceU16(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastU32(a, retVal []uint32, dimSize int, defaultValue uint32, fn func(a []uint32) uint32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastU32(a, retVal []uint32, dimSize int, defaultValue uint32, fn func(uint32, uint32) uint32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceU32(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastU64(a, retVal []uint64, dimSize int, defaultValue uint64, fn func(a []uint64) uint64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastU64(a, retVal []uint64, dimSize int, defaultValue uint64, fn func(uint64, uint64) uint64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceU64(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastUintptr(a, retVal []uintptr, dimSize int, defaultValue uintptr, fn func(a []uintptr) uintptr) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastUintptr(a, retVal []uintptr, dimSize int, defaultValue uintptr, fn func(uintptr, uintptr) uintptr) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceUintptr(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastF32(a, retVal []float32, dimSize int, defaultValue float32, fn func(a []float32) float32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastF32(a, retVal []float32, dimSize int, defaultValue float32, fn func(float32, float32) float32) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceF32(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastF64(a, retVal []float64, dimSize int, defaultValue float64, fn func(a []float64) float64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastF64(a, retVal []float64, dimSize int, defaultValue float64, fn func(float64, float64) float64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceF64(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastC64(a, retVal []complex64, dimSize int, defaultValue complex64, fn func(a []complex64) complex64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastC64(a, retVal []complex64, dimSize int, defaultValue complex64, fn func(complex64, complex64) complex64) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceC64(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastC128(a, retVal []complex128, dimSize int, defaultValue complex128, fn func(a []complex128) complex128) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastC128(a, retVal []complex128, dimSize int, defaultValue complex128, fn func(complex128, complex128) complex128) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceC128(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastStr(a, retVal []string, dimSize int, defaultValue string, fn func(a []string) string) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastStr(a, retVal []string, dimSize int, defaultValue string, fn func(string, string) string) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceStr(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceLastUnsafePointer(a, retVal []unsafe.Pointer, dimSize int, defaultValue unsafe.Pointer, fn func(a []unsafe.Pointer) unsafe.Pointer) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := fn(a[start : start+dimSize])
		retVal[at] = r
		at++
	}
}

func genericReduceLastUnsafePointer(a, retVal []unsafe.Pointer, dimSize int, defaultValue unsafe.Pointer, fn func(unsafe.Pointer, unsafe.Pointer) unsafe.Pointer) {
	var at int
	for start := 0; start <= len(a)-dimSize; start += dimSize {
		r := ReduceUnsafePointer(fn, defaultValue, a[start:start+dimSize]...)
		retVal[at] = r
		at++
	}
}

func reduceDefaultB(data, retVal []bool, dim0, dimSize, outerStride, stride, expected int, fn func(a, b bool) bool) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultI(data, retVal []int, dim0, dimSize, outerStride, stride, expected int, fn func(a, b int) int) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultI8(data, retVal []int8, dim0, dimSize, outerStride, stride, expected int, fn func(a, b int8) int8) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultI16(data, retVal []int16, dim0, dimSize, outerStride, stride, expected int, fn func(a, b int16) int16) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultI32(data, retVal []int32, dim0, dimSize, outerStride, stride, expected int, fn func(a, b int32) int32) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultI64(data, retVal []int64, dim0, dimSize, outerStride, stride, expected int, fn func(a, b int64) int64) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultU(data, retVal []uint, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uint) uint) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultU8(data, retVal []uint8, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uint8) uint8) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultU16(data, retVal []uint16, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uint16) uint16) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultU32(data, retVal []uint32, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uint32) uint32) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultU64(data, retVal []uint64, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uint64) uint64) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultUintptr(data, retVal []uintptr, dim0, dimSize, outerStride, stride, expected int, fn func(a, b uintptr) uintptr) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultF32(data, retVal []float32, dim0, dimSize, outerStride, stride, expected int, fn func(a, b float32) float32) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultF64(data, retVal []float64, dim0, dimSize, outerStride, stride, expected int, fn func(a, b float64) float64) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultC64(data, retVal []complex64, dim0, dimSize, outerStride, stride, expected int, fn func(a, b complex64) complex64) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultC128(data, retVal []complex128, dim0, dimSize, outerStride, stride, expected int, fn func(a, b complex128) complex128) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultStr(data, retVal []string, dim0, dimSize, outerStride, stride, expected int, fn func(a, b string) string) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}

func reduceDefaultUnsafePointer(data, retVal []unsafe.Pointer, dim0, dimSize, outerStride, stride, expected int, fn func(a, b unsafe.Pointer) unsafe.Pointer) {
	for i := 0; i < dim0; i++ {
		start := i * outerStride
		sliced := data[start : start+outerStride]
		var innerStart, strideTrack int
		for j := 0; j < expected; j++ {
			writeTo := i*expected + j
			retVal[writeTo] = sliced[innerStart]
			for k := 1; k < dimSize; k++ {
				readFrom := innerStart + k*stride
				retVal[writeTo] = fn(retVal[writeTo], sliced[readFrom])
			}
			strideTrack++
			if strideTrack >= stride {
				strideTrack = 0
				innerStart += stride
			}
			innerStart++
		}
	}
}
