package tensor

import (
	"testing"

	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestFromMat64(t *testing.T) {
	assert := assert.New(t)
	var m *mat64.Dense
	var T *Dense
	var backing []float64

	backing = Range(Float64, 0, 6).(f64s).Float64s()
	m = mat64.NewDense(2, 3, backing)

	T = FromMat64(m, true)
	assert.Equal(Shape{2, 3}, T.Shape())
	assert.Equal(backing, T.Data())
	backing[0] = 1000
	assert.NotEqual(backing, T.Data())

	backing[0] = 0
	T = FromMat64(m, false)
	backing[0] = 1000
	assert.Equal(backing, T.Data())
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
}
