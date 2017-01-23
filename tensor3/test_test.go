package tensor

import (
	"math/rand"
	"reflect"
)

/*
GENERATED FILE. DO NOT EDIT
*/

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
