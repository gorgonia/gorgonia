package tensor

import (
	"unsafe"

	"github.com/pkg/errors"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Eq */

func eqDDBoolsB(a, b []bool) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDSBoolsB(a []bool, b bool) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsC64(a, b []complex64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameC64(a, b []complex64) (retVal []complex64) {
	retVal = make([]complex64, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsC64(a []complex64, b complex64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameC64(a []complex64, b complex64) (retVal []complex64) {
	retVal = make([]complex64, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsC128(a, b []complex128) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDDSameC128(a, b []complex128) (retVal []complex128) {
	retVal = make([]complex128, len(a))
	for i, v := range a {
		if v == b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDSBoolsC128(a []complex128, b complex128) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDSSameC128(a []complex128, b complex128) (retVal []complex128) {
	retVal = make([]complex128, len(a))
	for i, v := range a {
		if v == b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func eqDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

func eqDDBoolsUnsafePointer(a, b []unsafe.Pointer) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b[i]
	}
	return retVal
}

func eqDSBoolsUnsafePointer(a []unsafe.Pointer, b unsafe.Pointer) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v == b
	}
	return retVal
}

/* Ne */

func neDDBoolsB(a, b []bool) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDSBoolsB(a []bool, b bool) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsC64(a, b []complex64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameC64(a, b []complex64) (retVal []complex64) {
	retVal = make([]complex64, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsC64(a []complex64, b complex64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameC64(a []complex64, b complex64) (retVal []complex64) {
	retVal = make([]complex64, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsC128(a, b []complex128) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDDSameC128(a, b []complex128) (retVal []complex128) {
	retVal = make([]complex128, len(a))
	for i, v := range a {
		if v != b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDSBoolsC128(a []complex128, b complex128) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDSSameC128(a []complex128, b complex128) (retVal []complex128) {
	retVal = make([]complex128, len(a))
	for i, v := range a {
		if v != b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func neDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

func neDDBoolsUnsafePointer(a, b []unsafe.Pointer) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b[i]
	}
	return retVal
}

func neDSBoolsUnsafePointer(a []unsafe.Pointer, b unsafe.Pointer) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v != b
	}
	return retVal
}

/* Gt */

func gtDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v > b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

func gtDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v > b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gtDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b[i]
	}
	return retVal
}

func gtDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v > b
	}
	return retVal
}

/* Gte */

func gteDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v >= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

func gteDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v >= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func gteDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b[i]
	}
	return retVal
}

func gteDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v >= b
	}
	return retVal
}

/* Lt */

func ltDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v < b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

func ltDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v < b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func ltDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b[i]
	}
	return retVal
}

func ltDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v < b
	}
	return retVal
}

/* Lte */

func lteDDBoolsI(a, b []int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameI(a, b []int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsI(a []int, b int) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameI(a []int, b int) (retVal []int) {
	retVal = make([]int, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsI8(a, b []int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameI8(a, b []int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsI8(a []int8, b int8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameI8(a []int8, b int8) (retVal []int8) {
	retVal = make([]int8, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsI16(a, b []int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameI16(a, b []int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsI16(a []int16, b int16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameI16(a []int16, b int16) (retVal []int16) {
	retVal = make([]int16, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsI32(a, b []int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameI32(a, b []int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsI32(a []int32, b int32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameI32(a []int32, b int32) (retVal []int32) {
	retVal = make([]int32, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsI64(a, b []int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameI64(a, b []int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsI64(a []int64, b int64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameI64(a []int64, b int64) (retVal []int64) {
	retVal = make([]int64, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsU(a, b []uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameU(a, b []uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsU(a []uint, b uint) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameU(a []uint, b uint) (retVal []uint) {
	retVal = make([]uint, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsU8(a, b []uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameU8(a, b []uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsU8(a []uint8, b uint8) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameU8(a []uint8, b uint8) (retVal []uint8) {
	retVal = make([]uint8, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsU16(a, b []uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameU16(a, b []uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsU16(a []uint16, b uint16) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameU16(a []uint16, b uint16) (retVal []uint16) {
	retVal = make([]uint16, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsU32(a, b []uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameU32(a, b []uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsU32(a []uint32, b uint32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameU32(a []uint32, b uint32) (retVal []uint32) {
	retVal = make([]uint32, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsU64(a, b []uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameU64(a, b []uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsU64(a []uint64, b uint64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameU64(a []uint64, b uint64) (retVal []uint64) {
	retVal = make([]uint64, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsUintptr(a, b []uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDSBoolsUintptr(a []uintptr, b uintptr) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDDBoolsF32(a, b []float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameF32(a, b []float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsF32(a []float32, b float32) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameF32(a []float32, b float32) (retVal []float32) {
	retVal = make([]float32, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsF64(a, b []float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDDSameF64(a, b []float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v <= b[i] {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDSBoolsF64(a []float64, b float64) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func lteDSSameF64(a []float64, b float64) (retVal []float64) {
	retVal = make([]float64, len(a))
	for i, v := range a {
		if v <= b {
			retVal[i] = 1
		} else {
			retVal[i] = 0
		}
	}
	return retVal
}

func lteDDBoolsStr(a, b []string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b[i]
	}
	return retVal
}

func lteDSBoolsStr(a []string, b string) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v <= b
	}
	return retVal
}

func vecMinI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxI(a, b []int) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxI8(a, b []int8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxI16(a, b []int16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxI32(a, b []int32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxI64(a, b []int64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxU(a, b []uint) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxU8(a, b []uint8) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxU16(a, b []uint16) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxU32(a, b []uint32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxU64(a, b []uint64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinUintptr(a, b []uintptr) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxUintptr(a, b []uintptr) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxF32(a, b []float32) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxF64(a, b []float64) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func vecMinStr(a, b []string) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func vecMaxStr(a, b []string) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
func minI(a, b int) (c int) {
	if a < b {
		return a
	}
	return b
}

func maxI(a, b int) (c int) {
	if a > b {
		return a
	}
	return b
}
func minI8(a, b int8) (c int8) {
	if a < b {
		return a
	}
	return b
}

func maxI8(a, b int8) (c int8) {
	if a > b {
		return a
	}
	return b
}
func minI16(a, b int16) (c int16) {
	if a < b {
		return a
	}
	return b
}

func maxI16(a, b int16) (c int16) {
	if a > b {
		return a
	}
	return b
}
func minI32(a, b int32) (c int32) {
	if a < b {
		return a
	}
	return b
}

func maxI32(a, b int32) (c int32) {
	if a > b {
		return a
	}
	return b
}
func minI64(a, b int64) (c int64) {
	if a < b {
		return a
	}
	return b
}

func maxI64(a, b int64) (c int64) {
	if a > b {
		return a
	}
	return b
}
func minU(a, b uint) (c uint) {
	if a < b {
		return a
	}
	return b
}

func maxU(a, b uint) (c uint) {
	if a > b {
		return a
	}
	return b
}
func minU8(a, b uint8) (c uint8) {
	if a < b {
		return a
	}
	return b
}

func maxU8(a, b uint8) (c uint8) {
	if a > b {
		return a
	}
	return b
}
func minU16(a, b uint16) (c uint16) {
	if a < b {
		return a
	}
	return b
}

func maxU16(a, b uint16) (c uint16) {
	if a > b {
		return a
	}
	return b
}
func minU32(a, b uint32) (c uint32) {
	if a < b {
		return a
	}
	return b
}

func maxU32(a, b uint32) (c uint32) {
	if a > b {
		return a
	}
	return b
}
func minU64(a, b uint64) (c uint64) {
	if a < b {
		return a
	}
	return b
}

func maxU64(a, b uint64) (c uint64) {
	if a > b {
		return a
	}
	return b
}
func minUintptr(a, b uintptr) (c uintptr) {
	if a < b {
		return a
	}
	return b
}

func maxUintptr(a, b uintptr) (c uintptr) {
	if a > b {
		return a
	}
	return b
}
func minF32(a, b float32) (c float32) {
	if a < b {
		return a
	}
	return b
}

func maxF32(a, b float32) (c float32) {
	if a > b {
		return a
	}
	return b
}
func minF64(a, b float64) (c float64) {
	if a < b {
		return a
	}
	return b
}

func maxF64(a, b float64) (c float64) {
	if a > b {
		return a
	}
	return b
}
func minStr(a, b string) (c string) {
	if a < b {
		return a
	}
	return b
}

func maxStr(a, b string) (c string) {
	if a > b {
		return a
	}
	return b
}
