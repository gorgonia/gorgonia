package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var addTests = []struct {
	a interface{}
	b interface{}

	correct interface{}
}{
	// Float64
	{float64(1), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{2, 3, 4, 5}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), float64(1), []float64{2, 3, 4, 5}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{2, 4, 6, 8}},

	// Float32
	{float32(1), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{2, 3, 4, 5}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), float32(1), []float32{2, 3, 4, 5}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{2, 4, 6, 8}},

	// Int
	{int(1), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{2, 3, 4, 5}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), int(1), []int{2, 3, 4, 5}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{2, 4, 6, 8}},

	// Int64
	{int64(1), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{2, 3, 4, 5}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), int64(1), []int64{2, 3, 4, 5}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{2, 4, 6, 8}},

	// Int32
	{int32(1), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{2, 3, 4, 5}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), int32(1), []int32{2, 3, 4, 5}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{2, 4, 6, 8}},

	// Byte
	{byte(1), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{2, 3, 4, 5}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), byte(1), []byte{2, 3, 4, 5}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{2, 4, 6, 8}},
}

func TestAdd(t *testing.T) {
	assert := assert.New(t)
	for _, ats := range addTests {
		T, err := Add(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct, T.Data())
	}
}

var subTests = []struct {
	a interface{}
	b interface{}

	correct interface{}
}{
	// Float64
	{float64(1), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{0, -1, -2, -3}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), float64(1), []float64{0, 1, 2, 3}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{0, 0, 0, 0}},

	// Float32
	{float32(1), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{0, -1, -2, -3}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), float32(1), []float32{0, 1, 2, 3}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{0, 0, 0, 0}},

	// Int
	{int(1), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{0, -1, -2, -3}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), int(1), []int{0, 1, 2, 3}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{0, 0, 0, 0}},

	// Int64
	{int64(1), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{0, -1, -2, -3}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), int64(1), []int64{0, 1, 2, 3}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{0, 0, 0, 0}},

	// Int32
	{int32(1), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{0, -1, -2, -3}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), int32(1), []int32{0, 1, 2, 3}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{0, 0, 0, 0}},

	// Byte
	{byte(1), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{0, 255, 254, 253}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), byte(1), []byte{0, 1, 2, 3}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{0, 0, 0, 0}},
}

func TestSub(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range subTests {
		T, err := Sub(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct, T.Data(), "Test %d", i)
	}
}

var mulTests = []struct {
	a interface{}
	b interface{}

	correct interface{}
}{
	// Float64
	{float64(1), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{1, 2, 3, 4}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), float64(1), []float64{1, 2, 3, 4}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{1, 4, 9, 16}},

	// Float32
	{float32(1), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{1, 2, 3, 4}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), float32(1), []float32{1, 2, 3, 4}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{1, 4, 9, 16}},

	// Int
	{int(1), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{1, 2, 3, 4}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), int(1), []int{1, 2, 3, 4}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{1, 4, 9, 16}},

	// Int64
	{int64(1), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{1, 2, 3, 4}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), int64(1), []int64{1, 2, 3, 4}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{1, 4, 9, 16}},

	// Int32
	{int32(1), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{1, 2, 3, 4}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), int32(1), []int32{1, 2, 3, 4}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{1, 4, 9, 16}},

	// Byte
	{byte(1), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{1, 2, 3, 4}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), byte(1), []byte{1, 2, 3, 4}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{1, 4, 9, 16}},
}

func TestMul(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range mulTests {
		T, err := Mul(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct, T.Data(), "Test %d", i)
	}
}

var divTests = []struct {
	a interface{}
	b interface{}

	correct interface{}
}{
	// Float64
	{float64(24), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{24, 12, 8, 6}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), float64(1), []float64{1, 2, 3, 4}},
	{New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})), []float64{1, 1, 1, 1}},

	// Float32
	{float32(24), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{24, 12, 8, 6}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), float32(1), []float32{1, 2, 3, 4}},
	{New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})), []float32{1, 1, 1, 1}},

	// Int
	{int(24), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{24, 12, 8, 6}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), int(1), []int{1, 2, 3, 4}},
	{New(Of(Int), WithBacking([]int{1, 2, 3, 4})), New(Of(Int), WithBacking([]int{1, 2, 3, 4})), []int{1, 1, 1, 1}},

	// Int64
	{int64(24), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{24, 12, 8, 6}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), int64(1), []int64{1, 2, 3, 4}},
	{New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})), []int64{1, 1, 1, 1}},

	// Int32
	{int32(24), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{24, 12, 8, 6}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), int32(1), []int32{1, 2, 3, 4}},
	{New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})), []int32{1, 1, 1, 1}},

	// Byte
	{byte(24), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{24, 12, 8, 6}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), byte(1), []byte{1, 2, 3, 4}},
	{New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})), []byte{1, 1, 1, 1}},
}

func TestDiv(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range divTests {
		T, err := Div(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct, T.Data(), "Test %d", i)
	}
}
