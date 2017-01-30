package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

var sumTests = []struct {
	name  string
	of    Dtype
	shape Shape
	along []int

	correctShape Shape
	correct      interface{}
}{
	{"common case: T.Sum() for int", Int, Shape{2, 3}, []int{}, ScalarShape(), int(15)},
	{"A.Sum(0) for int", Int, Shape{2, 3}, []int{0}, Shape{3}, []int{3, 5, 7}},
	{"A.Sum(1) for int", Int, Shape{2, 3}, []int{1}, Shape{2}, []int{3, 12}},
	{"A.Sum(0,1) for int", Int, Shape{2, 3}, []int{0, 1}, ScalarShape(), int(15)},
	{"A.Sum(1,0) for int", Int, Shape{2, 3}, []int{1, 0}, ScalarShape(), int(15)},
	{"3T.Sum(1,2) for int", Int, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []int{66, 210}},
	{"common case: T.Sum() for int8", Int8, Shape{2, 3}, []int{}, ScalarShape(), int8(15)},
	{"A.Sum(0) for int8", Int8, Shape{2, 3}, []int{0}, Shape{3}, []int8{3, 5, 7}},
	{"A.Sum(1) for int8", Int8, Shape{2, 3}, []int{1}, Shape{2}, []int8{3, 12}},
	{"A.Sum(0,1) for int8", Int8, Shape{2, 3}, []int{0, 1}, ScalarShape(), int8(15)},
	{"A.Sum(1,0) for int8", Int8, Shape{2, 3}, []int{1, 0}, ScalarShape(), int8(15)},
	{"3T.Sum(1,2) for int8", Int8, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []int8{66, -46}},
	{"common case: T.Sum() for int16", Int16, Shape{2, 3}, []int{}, ScalarShape(), int16(15)},
	{"A.Sum(0) for int16", Int16, Shape{2, 3}, []int{0}, Shape{3}, []int16{3, 5, 7}},
	{"A.Sum(1) for int16", Int16, Shape{2, 3}, []int{1}, Shape{2}, []int16{3, 12}},
	{"A.Sum(0,1) for int16", Int16, Shape{2, 3}, []int{0, 1}, ScalarShape(), int16(15)},
	{"A.Sum(1,0) for int16", Int16, Shape{2, 3}, []int{1, 0}, ScalarShape(), int16(15)},
	{"3T.Sum(1,2) for int16", Int16, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []int16{66, 210}},
	{"common case: T.Sum() for int32", Int32, Shape{2, 3}, []int{}, ScalarShape(), int32(15)},
	{"A.Sum(0) for int32", Int32, Shape{2, 3}, []int{0}, Shape{3}, []int32{3, 5, 7}},
	{"A.Sum(1) for int32", Int32, Shape{2, 3}, []int{1}, Shape{2}, []int32{3, 12}},
	{"A.Sum(0,1) for int32", Int32, Shape{2, 3}, []int{0, 1}, ScalarShape(), int32(15)},
	{"A.Sum(1,0) for int32", Int32, Shape{2, 3}, []int{1, 0}, ScalarShape(), int32(15)},
	{"3T.Sum(1,2) for int32", Int32, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []int32{66, 210}},
	{"common case: T.Sum() for int64", Int64, Shape{2, 3}, []int{}, ScalarShape(), int64(15)},
	{"A.Sum(0) for int64", Int64, Shape{2, 3}, []int{0}, Shape{3}, []int64{3, 5, 7}},
	{"A.Sum(1) for int64", Int64, Shape{2, 3}, []int{1}, Shape{2}, []int64{3, 12}},
	{"A.Sum(0,1) for int64", Int64, Shape{2, 3}, []int{0, 1}, ScalarShape(), int64(15)},
	{"A.Sum(1,0) for int64", Int64, Shape{2, 3}, []int{1, 0}, ScalarShape(), int64(15)},
	{"3T.Sum(1,2) for int64", Int64, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []int64{66, 210}},
	{"common case: T.Sum() for uint", Uint, Shape{2, 3}, []int{}, ScalarShape(), uint(15)},
	{"A.Sum(0) for uint", Uint, Shape{2, 3}, []int{0}, Shape{3}, []uint{3, 5, 7}},
	{"A.Sum(1) for uint", Uint, Shape{2, 3}, []int{1}, Shape{2}, []uint{3, 12}},
	{"A.Sum(0,1) for uint", Uint, Shape{2, 3}, []int{0, 1}, ScalarShape(), uint(15)},
	{"A.Sum(1,0) for uint", Uint, Shape{2, 3}, []int{1, 0}, ScalarShape(), uint(15)},
	{"3T.Sum(1,2) for uint", Uint, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []uint{66, 210}},
	{"common case: T.Sum() for uint8", Uint8, Shape{2, 3}, []int{}, ScalarShape(), uint8(15)},
	{"A.Sum(0) for uint8", Uint8, Shape{2, 3}, []int{0}, Shape{3}, []uint8{3, 5, 7}},
	{"A.Sum(1) for uint8", Uint8, Shape{2, 3}, []int{1}, Shape{2}, []uint8{3, 12}},
	{"A.Sum(0,1) for uint8", Uint8, Shape{2, 3}, []int{0, 1}, ScalarShape(), uint8(15)},
	{"A.Sum(1,0) for uint8", Uint8, Shape{2, 3}, []int{1, 0}, ScalarShape(), uint8(15)},
	{"3T.Sum(1,2) for uint8", Uint8, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []uint8{66, 210}},
	{"common case: T.Sum() for uint16", Uint16, Shape{2, 3}, []int{}, ScalarShape(), uint16(15)},
	{"A.Sum(0) for uint16", Uint16, Shape{2, 3}, []int{0}, Shape{3}, []uint16{3, 5, 7}},
	{"A.Sum(1) for uint16", Uint16, Shape{2, 3}, []int{1}, Shape{2}, []uint16{3, 12}},
	{"A.Sum(0,1) for uint16", Uint16, Shape{2, 3}, []int{0, 1}, ScalarShape(), uint16(15)},
	{"A.Sum(1,0) for uint16", Uint16, Shape{2, 3}, []int{1, 0}, ScalarShape(), uint16(15)},
	{"3T.Sum(1,2) for uint16", Uint16, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []uint16{66, 210}},
	{"common case: T.Sum() for uint32", Uint32, Shape{2, 3}, []int{}, ScalarShape(), uint32(15)},
	{"A.Sum(0) for uint32", Uint32, Shape{2, 3}, []int{0}, Shape{3}, []uint32{3, 5, 7}},
	{"A.Sum(1) for uint32", Uint32, Shape{2, 3}, []int{1}, Shape{2}, []uint32{3, 12}},
	{"A.Sum(0,1) for uint32", Uint32, Shape{2, 3}, []int{0, 1}, ScalarShape(), uint32(15)},
	{"A.Sum(1,0) for uint32", Uint32, Shape{2, 3}, []int{1, 0}, ScalarShape(), uint32(15)},
	{"3T.Sum(1,2) for uint32", Uint32, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []uint32{66, 210}},
	{"common case: T.Sum() for uint64", Uint64, Shape{2, 3}, []int{}, ScalarShape(), uint64(15)},
	{"A.Sum(0) for uint64", Uint64, Shape{2, 3}, []int{0}, Shape{3}, []uint64{3, 5, 7}},
	{"A.Sum(1) for uint64", Uint64, Shape{2, 3}, []int{1}, Shape{2}, []uint64{3, 12}},
	{"A.Sum(0,1) for uint64", Uint64, Shape{2, 3}, []int{0, 1}, ScalarShape(), uint64(15)},
	{"A.Sum(1,0) for uint64", Uint64, Shape{2, 3}, []int{1, 0}, ScalarShape(), uint64(15)},
	{"3T.Sum(1,2) for uint64", Uint64, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []uint64{66, 210}},
	{"common case: T.Sum() for float32", Float32, Shape{2, 3}, []int{}, ScalarShape(), float32(15)},
	{"A.Sum(0) for float32", Float32, Shape{2, 3}, []int{0}, Shape{3}, []float32{3, 5, 7}},
	{"A.Sum(1) for float32", Float32, Shape{2, 3}, []int{1}, Shape{2}, []float32{3, 12}},
	{"A.Sum(0,1) for float32", Float32, Shape{2, 3}, []int{0, 1}, ScalarShape(), float32(15)},
	{"A.Sum(1,0) for float32", Float32, Shape{2, 3}, []int{1, 0}, ScalarShape(), float32(15)},
	{"3T.Sum(1,2) for float32", Float32, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []float32{66, 210}},
	{"common case: T.Sum() for float64", Float64, Shape{2, 3}, []int{}, ScalarShape(), float64(15)},
	{"A.Sum(0) for float64", Float64, Shape{2, 3}, []int{0}, Shape{3}, []float64{3, 5, 7}},
	{"A.Sum(1) for float64", Float64, Shape{2, 3}, []int{1}, Shape{2}, []float64{3, 12}},
	{"A.Sum(0,1) for float64", Float64, Shape{2, 3}, []int{0, 1}, ScalarShape(), float64(15)},
	{"A.Sum(1,0) for float64", Float64, Shape{2, 3}, []int{1, 0}, ScalarShape(), float64(15)},
	{"3T.Sum(1,2) for float64", Float64, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []float64{66, 210}},
	{"common case: T.Sum() for complex64", Complex64, Shape{2, 3}, []int{}, ScalarShape(), complex64(15)},
	{"A.Sum(0) for complex64", Complex64, Shape{2, 3}, []int{0}, Shape{3}, []complex64{3, 5, 7}},
	{"A.Sum(1) for complex64", Complex64, Shape{2, 3}, []int{1}, Shape{2}, []complex64{3, 12}},
	{"A.Sum(0,1) for complex64", Complex64, Shape{2, 3}, []int{0, 1}, ScalarShape(), complex64(15)},
	{"A.Sum(1,0) for complex64", Complex64, Shape{2, 3}, []int{1, 0}, ScalarShape(), complex64(15)},
	{"3T.Sum(1,2) for complex64", Complex64, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []complex64{66, 210}},
	{"common case: T.Sum() for complex128", Complex128, Shape{2, 3}, []int{}, ScalarShape(), complex128(15)},
	{"A.Sum(0) for complex128", Complex128, Shape{2, 3}, []int{0}, Shape{3}, []complex128{3, 5, 7}},
	{"A.Sum(1) for complex128", Complex128, Shape{2, 3}, []int{1}, Shape{2}, []complex128{3, 12}},
	{"A.Sum(0,1) for complex128", Complex128, Shape{2, 3}, []int{0, 1}, ScalarShape(), complex128(15)},
	{"A.Sum(1,0) for complex128", Complex128, Shape{2, 3}, []int{1, 0}, ScalarShape(), complex128(15)},
	{"3T.Sum(1,2) for complex128", Complex128, Shape{2, 3, 4}, []int{1, 2}, Shape{2}, []complex128{66, 210}},
}

func TestDense_Sum(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Dense
	var err error

	for _, sts := range sumTests {
		T = New(WithShape(sts.shape...), WithBacking(Range(sts.of, 0, sts.shape.TotalSize())))
		if T2, err = T.Sum(sts.along...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()))
		assert.Equal(sts.correct, T2.Data())
	}

	// idiots
	_, err = T.Sum(1000)
	assert.NotNil(err)
}
