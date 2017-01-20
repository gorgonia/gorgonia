package tensor

import (
	"math"
	"testing"

	"github.com/chewxy/math32"
	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

var denseFromMat64 = []struct {
	of  Dtype
	mat []float64

	correct interface{}
}{
	{Float64, []float64{0, 1, 2, 3, 4, 5}, []float64{0, 1, 2, 3, 4, 5}},
	{Float32, []float64{0, 1, 2, 3, 4, 5}, []float32{0, 1, 2, 3, 4, 5}},
	{Int, []float64{0, 1, 2, math.NaN(), math.Inf(1), math.Inf(-1)}, []int{0, 1, 2, 0, 0, 0}},
	{Int64, []float64{0, 1, 2, math.NaN(), math.Inf(1), math.Inf(-1)}, []int64{0, 1, 2, 0, 0, 0}},
	{Int32, []float64{0, 1, 2, math.NaN(), math.Inf(1), math.Inf(-1)}, []int32{0, 1, 2, 0, 0, 0}},
	{Byte, []float64{0, 1, 2, math.NaN(), math.Inf(1), math.Inf(-1)}, []byte{0, 1, 2, 0, 0, 0}},
	{Bool, []float64{0, 1, 2, math.NaN(), math.Inf(1), math.Inf(-1)}, []bool{false, true, true, false, false, false}},
}

func TestFromMat64(t *testing.T) {
	assert := assert.New(t)
	for _, m64t := range denseFromMat64 {
		m := mat64.NewDense(2, 3, m64t.mat)
		T := FromMat64(m, m64t.of, true)
		assert.True(Shape{2, 3}.Eq(T.Shape()))
		assert.Equal(m64t.correct, T.Data())

		T = FromMat64(m, m64t.of, false)
		assert.True(Shape{2, 3}.Eq(T.Shape()))
		assert.Equal(m64t.correct, T.Data())
	}

	// test specials
	backing := []float64{math.NaN(), math.Inf(1), math.Inf(-1), math.NaN()}
	m := mat64.NewDense(2, 2, backing)

	T := FromMat64(m, Float64, true)
	assert.True(math.IsNaN(T.Data().([]float64)[0]))
	assert.True(math.IsInf(T.Data().([]float64)[1], 1))
	assert.True(math.IsInf(T.Data().([]float64)[2], -1))

	T = FromMat64(m, Float32, true)
	assert.True(math32.IsNaN(T.Data().([]float32)[0]))
	assert.True(math32.IsInf(T.Data().([]float32)[1], 1))
	assert.True(math32.IsInf(T.Data().([]float32)[2], -1))
}

var denseToMat64Tests = []struct {
	a      Array
	shape  Shape
	slices []Slice

	correct       []float64
	slicedCorrect []float64
	err           bool
}{
	{Range(Float64, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, nil, []float64{1000, 1, 3, 4}, false},
	{Range(Float32, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, []float64{0, 1, 3, 4}, false},
	{Range(Int, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, []float64{0, 1, 3, 4}, false},
	{Range(Int64, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, []float64{0, 1, 3, 4}, false},
	{Range(Int32, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, []float64{0, 1, 3, 4}, false},
	{Range(Byte, 0, 9), Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{0, 1, 2, 3, 4, 5, 6, 7, 8}, []float64{0, 1, 3, 4}, false},
	{bs{true, false, true, false, true, false, true, false, true}, Shape{3, 3}, []Slice{makeRS(0, 2), makeRS(0, 2)}, []float64{1, 0, 1, 0, 1, 0, 1, 0, 1}, []float64{1, 0, 0, 1}, false},

	// bad shape
	{Range(Float64, 0, 24), Shape{2, 3, 4}, nil, nil, nil, true},
}

func TestToMat64(t *testing.T) {
	assert := assert.New(t)
	for i, m64t := range denseToMat64Tests {
		T := New(WithBacking(m64t.a), WithShape(m64t.shape...))
		m, err := ToMat64(T, true)
		if checkErr(t, m64t.err, err, "ToMat64 Copy", i) {
			continue
		}

		if f, ok := m64t.a.(f64s); ok {
			assert.Equal(m64t.a.Data(), m.RawMatrix().Data)
			f[0] = 1000
			assert.NotEqual(m64t.a.Data(), m.RawMatrix().Data)
		} else {
			assert.Equal(m64t.correct, m.RawMatrix().Data)
		}

		// don't copy
		m, err = ToMat64(T, false)
		if checkErr(t, m64t.err, err, "ToMat64 NoCopy", i) {
			continue
		}
		if f, ok := m64t.a.(f64s); ok {
			assert.Equal(m64t.a.Data(), m.RawMatrix().Data)
			f[0] = 1000
			assert.Equal(m64t.a.Data(), m.RawMatrix().Data)
		} else {
			assert.Equal(m64t.correct, m.RawMatrix().Data)
		}

		T2, err := T.Slice(m64t.slices...)
		if err != nil {
			t.Error(err)
			continue
		}

		m, err = ToMat64(T2, true)
		if checkErr(t, m64t.err, err, "ToMat64 Sliced", i) {
			continue
		}

		assert.Equal(m64t.slicedCorrect, m.RawMatrix().Data)
	}

	backing := []float32{math32.NaN(), math32.Inf(1), math32.Inf(-1), math32.NaN()}
	T := New(WithBacking(backing), WithShape(2, 2))
	m, _ := ToMat64(T, false)
	data := m.RawMatrix().Data

	assert.True(math.IsNaN(data[0]))
	assert.True(math.IsInf(data[1], 1))
	assert.True(math.IsInf(data[2], -1))
}
