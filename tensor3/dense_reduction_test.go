package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

var denseReductionTests = []struct {
	of   Dtype
	fn   interface{}
	def  interface{}
	axis int

	correct      interface{}
	correctShape Shape
}{
	// int
	{Int, addI, int(0), 0, []int{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int, addI, int(0), 1, []int{6, 9, 24, 27}, Shape{2, 2}},
	{Int, addI, int(0), 2, []int{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int8
	{Int8, addI8, int8(0), 0, []int8{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int8, addI8, int8(0), 1, []int8{6, 9, 24, 27}, Shape{2, 2}},
	{Int8, addI8, int8(0), 2, []int8{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int16
	{Int16, addI16, int16(0), 0, []int16{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int16, addI16, int16(0), 1, []int16{6, 9, 24, 27}, Shape{2, 2}},
	{Int16, addI16, int16(0), 2, []int16{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int32
	{Int32, addI32, int32(0), 0, []int32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int32, addI32, int32(0), 1, []int32{6, 9, 24, 27}, Shape{2, 2}},
	{Int32, addI32, int32(0), 2, []int32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int64
	{Int64, addI64, int64(0), 0, []int64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int64, addI64, int64(0), 1, []int64{6, 9, 24, 27}, Shape{2, 2}},
	{Int64, addI64, int64(0), 2, []int64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint
	{Uint, addU, uint(0), 0, []uint{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint, addU, uint(0), 1, []uint{6, 9, 24, 27}, Shape{2, 2}},
	{Uint, addU, uint(0), 2, []uint{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint8
	{Uint8, addU8, uint8(0), 0, []uint8{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint8, addU8, uint8(0), 1, []uint8{6, 9, 24, 27}, Shape{2, 2}},
	{Uint8, addU8, uint8(0), 2, []uint8{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint16
	{Uint16, addU16, uint16(0), 0, []uint16{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint16, addU16, uint16(0), 1, []uint16{6, 9, 24, 27}, Shape{2, 2}},
	{Uint16, addU16, uint16(0), 2, []uint16{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint32
	{Uint32, addU32, uint32(0), 0, []uint32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint32, addU32, uint32(0), 1, []uint32{6, 9, 24, 27}, Shape{2, 2}},
	{Uint32, addU32, uint32(0), 2, []uint32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint64
	{Uint64, addU64, uint64(0), 0, []uint64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint64, addU64, uint64(0), 1, []uint64{6, 9, 24, 27}, Shape{2, 2}},
	{Uint64, addU64, uint64(0), 2, []uint64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// float32
	{Float32, addF32, float32(0), 0, []float32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Float32, addF32, float32(0), 1, []float32{6, 9, 24, 27}, Shape{2, 2}},
	{Float32, addF32, float32(0), 2, []float32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// float64
	{Float64, addF64, float64(0), 0, []float64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Float64, addF64, float64(0), 1, []float64{6, 9, 24, 27}, Shape{2, 2}},
	{Float64, addF64, float64(0), 2, []float64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// complex64
	{Complex64, addC64, complex64(0), 0, []complex64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Complex64, addC64, complex64(0), 1, []complex64{6, 9, 24, 27}, Shape{2, 2}},
	{Complex64, addC64, complex64(0), 2, []complex64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// complex128
	{Complex128, addC128, complex128(0), 0, []complex128{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Complex128, addC128, complex128(0), 1, []complex128{6, 9, 24, 27}, Shape{2, 2}},
	{Complex128, addC128, complex128(0), 2, []complex128{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
}

func TestDense_Reduce(t *testing.T) {
	assert := assert.New(t)
	for _, drt := range denseReductionTests {
		T := New(WithShape(2, 3, 2), WithBacking(Range(drt.of, 0, 2*3*2)))
		T2, err := T.Reduce(drt.fn, drt.def, drt.axis)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(drt.correctShape.Eq(T2.Shape()))
		assert.Equal(drt.correct, T2.Data())
	}
}
