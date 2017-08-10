package tensor

import (
	"math"
	"math/cmplx"
	"math/rand"
	"reflect"
	"unsafe"

	"github.com/chewxy/math32"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func anyToFloat64s(x interface{}) (retVal []float64) {
	switch xt := x.(type) {
	case []int:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int8:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int16:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []int64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint8:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint16:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []uint64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			retVal[i] = float64(v)
		}
		return
	case []float32:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case math32.IsNaN(v):
				retVal[i] = math.NaN()
			case math32.IsInf(v, 1):
				retVal[i] = math.Inf(1)
			case math32.IsInf(v, -1):
				retVal[i] = math.Inf(-1)
			default:
				retVal[i] = float64(v)
			}
		}
		return
	case []float64:
		return xt
	case []complex64:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case cmplx.IsNaN(complex128(v)):
				retVal[i] = math.NaN()
			case cmplx.IsInf(complex128(v)):
				retVal[i] = math.Inf(1)
			default:
				retVal[i] = float64(real(v))
			}
		}
		return
	case []complex128:
		retVal = make([]float64, len(xt))
		for i, v := range xt {
			switch {
			case cmplx.IsNaN(v):
				retVal[i] = math.NaN()
			case cmplx.IsInf(v):
				retVal[i] = math.Inf(1)
			default:
				retVal[i] = real(v)
			}
		}
		return
	}
	panic("Unreachable")
}

func identityB(a bool) bool                                 { return a }
func identityI(a int) int                                   { return a }
func identityI8(a int8) int8                                { return a }
func identityI16(a int16) int16                             { return a }
func identityI32(a int32) int32                             { return a }
func identityI64(a int64) int64                             { return a }
func identityU(a uint) uint                                 { return a }
func identityU8(a uint8) uint8                              { return a }
func identityU16(a uint16) uint16                           { return a }
func identityU32(a uint32) uint32                           { return a }
func identityU64(a uint64) uint64                           { return a }
func identityUintptr(a uintptr) uintptr                     { return a }
func identityF32(a float32) float32                         { return a }
func identityF64(a float64) float64                         { return a }
func identityC64(a complex64) complex64                     { return a }
func identityC128(a complex128) complex128                  { return a }
func identityStr(a string) string                           { return a }
func identityUnsafePointer(a unsafe.Pointer) unsafe.Pointer { return a }
func mutateB(a bool) bool                                   { return true }
func mutateI(a int) int                                     { return 1 }
func mutateI8(a int8) int8                                  { return 1 }
func mutateI16(a int16) int16                               { return 1 }
func mutateI32(a int32) int32                               { return 1 }
func mutateI64(a int64) int64                               { return 1 }
func mutateU(a uint) uint                                   { return 1 }
func mutateU8(a uint8) uint8                                { return 1 }
func mutateU16(a uint16) uint16                             { return 1 }
func mutateU32(a uint32) uint32                             { return 1 }
func mutateU64(a uint64) uint64                             { return 1 }
func mutateUintptr(a uintptr) uintptr                       { return 0xdeadbeef }
func mutateF32(a float32) float32                           { return 1 }
func mutateF64(a float64) float64                           { return 1 }
func mutateC64(a complex64) complex64                       { return 1 }
func mutateC128(a complex128) complex128                    { return 1 }
func mutateStr(a string) string                             { return "Hello World" }
func mutateUnsafePointer(a unsafe.Pointer) unsafe.Pointer   { return unsafe.Pointer(uintptr(0xdeadbeef)) }

type QCDenseB struct {
	*Dense
}

func (*QCDenseB) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]bool, size)
	for i := range s {
		s[i] = randomBool()
	}
	d := recycledDense(Bool, Shape{size}, WithBacking(s))
	q := new(QCDenseB)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseI struct {
	*Dense
}

func (*QCDenseI) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]int, size)
	for i := range s {
		s[i] = int(r.Int())
	}
	d := recycledDense(Int, Shape{size}, WithBacking(s))
	q := new(QCDenseI)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseI8 struct {
	*Dense
}

func (*QCDenseI8) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]int8, size)
	for i := range s {
		s[i] = int8(r.Int())
	}
	d := recycledDense(Int8, Shape{size}, WithBacking(s))
	q := new(QCDenseI8)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseI16 struct {
	*Dense
}

func (*QCDenseI16) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]int16, size)
	for i := range s {
		s[i] = int16(r.Int())
	}
	d := recycledDense(Int16, Shape{size}, WithBacking(s))
	q := new(QCDenseI16)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseI32 struct {
	*Dense
}

func (*QCDenseI32) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]int32, size)
	for i := range s {
		s[i] = int32(r.Int())
	}
	d := recycledDense(Int32, Shape{size}, WithBacking(s))
	q := new(QCDenseI32)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseI64 struct {
	*Dense
}

func (*QCDenseI64) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]int64, size)
	for i := range s {
		s[i] = int64(r.Int())
	}
	d := recycledDense(Int64, Shape{size}, WithBacking(s))
	q := new(QCDenseI64)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseU struct {
	*Dense
}

func (*QCDenseU) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uint, size)
	for i := range s {
		s[i] = uint(r.Uint32())
	}
	d := recycledDense(Uint, Shape{size}, WithBacking(s))
	q := new(QCDenseU)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseU8 struct {
	*Dense
}

func (*QCDenseU8) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uint8, size)
	for i := range s {
		s[i] = uint8(r.Uint32())
	}
	d := recycledDense(Uint8, Shape{size}, WithBacking(s))
	q := new(QCDenseU8)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseU16 struct {
	*Dense
}

func (*QCDenseU16) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uint16, size)
	for i := range s {
		s[i] = uint16(r.Uint32())
	}
	d := recycledDense(Uint16, Shape{size}, WithBacking(s))
	q := new(QCDenseU16)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseU32 struct {
	*Dense
}

func (*QCDenseU32) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uint32, size)
	for i := range s {
		s[i] = uint32(r.Uint32())
	}
	d := recycledDense(Uint32, Shape{size}, WithBacking(s))
	q := new(QCDenseU32)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseU64 struct {
	*Dense
}

func (*QCDenseU64) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uint64, size)
	for i := range s {
		s[i] = uint64(r.Uint32())
	}
	d := recycledDense(Uint64, Shape{size}, WithBacking(s))
	q := new(QCDenseU64)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseUintptr struct {
	*Dense
}

func (*QCDenseUintptr) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]uintptr, size)
	for i := range s {
		s[i] = uintptr(r.Uint32())
	}
	d := recycledDense(Uintptr, Shape{size}, WithBacking(s))
	q := new(QCDenseUintptr)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseF32 struct {
	*Dense
}

func (*QCDenseF32) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]float32, size)
	for i := range s {
		s[i] = r.Float32()
	}
	d := recycledDense(Float32, Shape{size}, WithBacking(s))
	q := new(QCDenseF32)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseF64 struct {
	*Dense
}

func (*QCDenseF64) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]float64, size)
	for i := range s {
		s[i] = r.Float64()
	}
	d := recycledDense(Float64, Shape{size}, WithBacking(s))
	q := new(QCDenseF64)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseC64 struct {
	*Dense
}

func (*QCDenseC64) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]complex64, size)
	for i := range s {
		s[i] = complex(r.Float32(), r.Float32())
	}
	d := recycledDense(Complex64, Shape{size}, WithBacking(s))
	q := new(QCDenseC64)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseC128 struct {
	*Dense
}

func (*QCDenseC128) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]complex128, size)
	for i := range s {
		s[i] = complex(r.Float64(), r.Float64())
	}
	d := recycledDense(Complex128, Shape{size}, WithBacking(s))
	q := new(QCDenseC128)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseStr struct {
	*Dense
}

func (*QCDenseStr) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]string, size)
	for i := range s {
		s[i] = randomString()
	}
	d := recycledDense(String, Shape{size}, WithBacking(s))
	q := new(QCDenseStr)
	q.Dense = d
	return reflect.ValueOf(q)
}

type QCDenseUnsafePointer struct {
	*Dense
}

func (*QCDenseUnsafePointer) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]unsafe.Pointer, size)
	for i := range s {
		s[i] = nil
	}
	d := recycledDense(UnsafePointer, Shape{size}, WithBacking(s))
	q := new(QCDenseUnsafePointer)
	q.Dense = d
	return reflect.ValueOf(q)
}
