package tensor

import (
	"math/rand"
	"reflect"
	"time"
	"unsafe"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func randomBool() bool {
	i := rand.Intn(11)
	return i > 5
}

// from : https://stackoverflow.com/a/31832326/3426066
const letterBytes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
const (
	letterIdxBits = 6                    // 6 bits to represent a letter index
	letterIdxMask = 1<<letterIdxBits - 1 // All 1-bits, as many as letterIdxBits
	letterIdxMax  = 63 / letterIdxBits   // # of letter indices fitting in 63 bits
)

var src = rand.NewSource(time.Now().UnixNano())

func randomString() string {
	n := rand.Intn(10)
	b := make([]byte, n)
	// A src.Int63() generates 63 random bits, enough for letterIdxMax characters!
	for i, cache, remain := n-1, src.Int63(), letterIdxMax; i >= 0; {
		if remain == 0 {
			cache, remain = src.Int63(), letterIdxMax
		}
		if idx := int(cache & letterIdxMask); idx < len(letterBytes) {
			b[i] = letterBytes[idx]
			i--
		}
		cache >>= letterIdxBits
		remain--
	}

	return string(b)
}

type QCDenseB struct {
	*Dense
}

func (q *QCDenseB) D() *Dense { return q.Dense }
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

func (q *QCDenseI) D() *Dense { return q.Dense }
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

func (q *QCDenseI8) D() *Dense { return q.Dense }
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

func (q *QCDenseI16) D() *Dense { return q.Dense }
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

func (q *QCDenseI32) D() *Dense { return q.Dense }
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

func (q *QCDenseI64) D() *Dense { return q.Dense }
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

func (q *QCDenseU) D() *Dense { return q.Dense }
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

func (q *QCDenseU8) D() *Dense { return q.Dense }
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

func (q *QCDenseU16) D() *Dense { return q.Dense }
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

func (q *QCDenseU32) D() *Dense { return q.Dense }
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

func (q *QCDenseU64) D() *Dense { return q.Dense }
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

func (q *QCDenseUintptr) D() *Dense { return q.Dense }
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

func (q *QCDenseF32) D() *Dense { return q.Dense }
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

func (q *QCDenseF64) D() *Dense { return q.Dense }
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

func (q *QCDenseC64) D() *Dense { return q.Dense }
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

func (q *QCDenseC128) D() *Dense { return q.Dense }
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

func (q *QCDenseStr) D() *Dense { return q.Dense }
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

func (q *QCDenseUnsafePointer) D() *Dense { return q.Dense }
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
