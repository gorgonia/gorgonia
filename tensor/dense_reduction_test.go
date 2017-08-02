package tensor

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
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
	{Int, execution.AddI, int(0), 0, []int{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int, execution.AddI, int(0), 1, []int{6, 9, 24, 27}, Shape{2, 2}},
	{Int, execution.AddI, int(0), 2, []int{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int8
	{Int8, execution.AddI8, int8(0), 0, []int8{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int8, execution.AddI8, int8(0), 1, []int8{6, 9, 24, 27}, Shape{2, 2}},
	{Int8, execution.AddI8, int8(0), 2, []int8{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int16
	{Int16, execution.AddI16, int16(0), 0, []int16{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int16, execution.AddI16, int16(0), 1, []int16{6, 9, 24, 27}, Shape{2, 2}},
	{Int16, execution.AddI16, int16(0), 2, []int16{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int32
	{Int32, execution.AddI32, int32(0), 0, []int32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int32, execution.AddI32, int32(0), 1, []int32{6, 9, 24, 27}, Shape{2, 2}},
	{Int32, execution.AddI32, int32(0), 2, []int32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// int64
	{Int64, execution.AddI64, int64(0), 0, []int64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Int64, execution.AddI64, int64(0), 1, []int64{6, 9, 24, 27}, Shape{2, 2}},
	{Int64, execution.AddI64, int64(0), 2, []int64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint
	{Uint, execution.AddU, uint(0), 0, []uint{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint, execution.AddU, uint(0), 1, []uint{6, 9, 24, 27}, Shape{2, 2}},
	{Uint, execution.AddU, uint(0), 2, []uint{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint8
	{Uint8, execution.AddU8, uint8(0), 0, []uint8{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint8, execution.AddU8, uint8(0), 1, []uint8{6, 9, 24, 27}, Shape{2, 2}},
	{Uint8, execution.AddU8, uint8(0), 2, []uint8{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint16
	{Uint16, execution.AddU16, uint16(0), 0, []uint16{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint16, execution.AddU16, uint16(0), 1, []uint16{6, 9, 24, 27}, Shape{2, 2}},
	{Uint16, execution.AddU16, uint16(0), 2, []uint16{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint32
	{Uint32, execution.AddU32, uint32(0), 0, []uint32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint32, execution.AddU32, uint32(0), 1, []uint32{6, 9, 24, 27}, Shape{2, 2}},
	{Uint32, execution.AddU32, uint32(0), 2, []uint32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// uint64
	{Uint64, execution.AddU64, uint64(0), 0, []uint64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Uint64, execution.AddU64, uint64(0), 1, []uint64{6, 9, 24, 27}, Shape{2, 2}},
	{Uint64, execution.AddU64, uint64(0), 2, []uint64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// float32
	{Float32, execution.AddF32, float32(0), 0, []float32{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Float32, execution.AddF32, float32(0), 1, []float32{6, 9, 24, 27}, Shape{2, 2}},
	{Float32, execution.AddF32, float32(0), 2, []float32{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// float64
	{Float64, execution.AddF64, float64(0), 0, []float64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Float64, execution.AddF64, float64(0), 1, []float64{6, 9, 24, 27}, Shape{2, 2}},
	{Float64, execution.AddF64, float64(0), 2, []float64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// complex64
	{Complex64, execution.AddC64, complex64(0), 0, []complex64{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Complex64, execution.AddC64, complex64(0), 1, []complex64{6, 9, 24, 27}, Shape{2, 2}},
	{Complex64, execution.AddC64, complex64(0), 2, []complex64{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
	// complex128
	{Complex128, execution.AddC128, complex128(0), 0, []complex128{6, 8, 10, 12, 14, 16}, Shape{3, 2}},
	{Complex128, execution.AddC128, complex128(0), 1, []complex128{6, 9, 24, 27}, Shape{2, 2}},
	{Complex128, execution.AddC128, complex128(0), 2, []complex128{1, 5, 9, 13, 17, 21}, Shape{2, 3}},
}

func TestDense_Reduce(t *testing.T) {
	assert := assert.New(t)
	for _, drt := range denseReductionTests {
		T := New(WithShape(2, 3, 2), WithBacking(Range(drt.of, 0, 2*3*2)))
		// T2, err := T.Reduce(drt.fn, drt.def, drt.axis)
		T2, err := T.Reduce(drt.fn, drt.axis)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(drt.correctShape.Eq(T2.Shape()))
		assert.Equal(drt.correct, T2.Data())

		// stupids:
		// _, err = T.Reduce(drt.fn, drt.def, 1000)
		_, err = T.Reduce(drt.fn, 1000)
		assert.NotNil(err)

		// wrong function type
		var f interface{}
		f = func(a, b float64) float64 { return 0 }
		if drt.of == Float64 {
			f = func(a, b int) int { return 0 }
		}

		// _, err = T.Reduce(f, drt.correct, 0)
		_, err = T.Reduce(f, 0)
		assert.NotNil(err)

		// wrong default value type
		// var def2 interface{}
		// def2 = 3.14
		// if drt.of == Float64 {
		// 	def2 = int(1)
		// }

		// _, err = T.Reduce(drt.fn, def2, 3) // only last axis requires a default value
		_, err = T.Reduce(drt.fn, 3)
		assert.NotNil(err)
	}
}
