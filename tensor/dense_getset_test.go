package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

var denseSetGetTests = []struct {
	of   Dtype
	data interface{}

	correct []interface{}
}{
	{Bool, []bool{true, false, true, false, true}, []interface{}{bool(true), bool(false), bool(true), bool(false), bool(true)}},
	{Int, []int{0, 1, 2, 3, 4}, []interface{}{int(0), int(1), int(2), int(3), int(4)}},
	{Int8, []int8{0, 1, 2, 3, 4}, []interface{}{int8(0), int8(1), int8(2), int8(3), int8(4)}},
	{Int16, []int16{0, 1, 2, 3, 4}, []interface{}{int16(0), int16(1), int16(2), int16(3), int16(4)}},
	{Int32, []int32{0, 1, 2, 3, 4}, []interface{}{int32(0), int32(1), int32(2), int32(3), int32(4)}},
	{Int64, []int64{0, 1, 2, 3, 4}, []interface{}{int64(0), int64(1), int64(2), int64(3), int64(4)}},
	{Uint, []uint{0, 1, 2, 3, 4}, []interface{}{uint(0), uint(1), uint(2), uint(3), uint(4)}},
	{Uint8, []uint8{0, 1, 2, 3, 4}, []interface{}{uint8(0), uint8(1), uint8(2), uint8(3), uint8(4)}},
	{Uint16, []uint16{0, 1, 2, 3, 4}, []interface{}{uint16(0), uint16(1), uint16(2), uint16(3), uint16(4)}},
	{Uint32, []uint32{0, 1, 2, 3, 4}, []interface{}{uint32(0), uint32(1), uint32(2), uint32(3), uint32(4)}},
	{Uint64, []uint64{0, 1, 2, 3, 4}, []interface{}{uint64(0), uint64(1), uint64(2), uint64(3), uint64(4)}},
	{Float32, []float32{0, 1, 2, 3, 4}, []interface{}{float32(0), float32(1), float32(2), float32(3), float32(4)}},
	{Float64, []float64{0, 1, 2, 3, 4}, []interface{}{float64(0), float64(1), float64(2), float64(3), float64(4)}},
	{Complex64, []complex64{0, 1, 2, 3, 4}, []interface{}{complex64(0), complex64(1), complex64(2), complex64(3), complex64(4)}},
	{Complex128, []complex128{0, 1, 2, 3, 4}, []interface{}{complex128(0), complex128(1), complex128(2), complex128(3), complex128(4)}},
	{String, []string{"zero", "one", "two", "three", "four"}, []interface{}{string("zero"), string("one"), string("two"), string("three"), string("four")}},
}

func TestDense_setget(t *testing.T) {
	assert := assert.New(t)
	for _, gts := range denseSetGetTests {
		T := New(Of(gts.of), WithShape(len(gts.correct)))
		for i, v := range gts.correct {
			T.set(i, v)
			got := T.get(i)
			assert.Equal(v, got)
		}
	}
}
