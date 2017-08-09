package tensor

import (
	"testing"

	"github.com/chewxy/vecf64"
	"github.com/stretchr/testify/assert"
)

type linalgTest struct {
	a, b           interface{}
	shapeA, shapeB Shape

	reuse, incr    interface{}
	shapeR, shapeI Shape

	correct          interface{}
	correctIncr      interface{}
	correctIncrReuse interface{}
	correctShape     Shape
	err              bool
	errIncr          bool
	errReuse         bool
}

var traceTests = []struct {
	data interface{}

	correct interface{}
	err     bool
}{
	{[]int{0,1,2,3,4,5}, int(4), false},
{[]int8{0,1,2,3,4,5}, int8(4), false},
{[]int16{0,1,2,3,4,5}, int16(4), false},
{[]int32{0,1,2,3,4,5}, int32(4), false},
{[]int64{0,1,2,3,4,5}, int64(4), false},
{[]uint{0,1,2,3,4,5}, uint(4), false},
{[]uint8{0,1,2,3,4,5}, uint8(4), false},
{[]uint16{0,1,2,3,4,5}, uint16(4), false},
{[]uint32{0,1,2,3,4,5}, uint32(4), false},
{[]uint64{0,1,2,3,4,5}, uint64(4), false},
{[]float32{0,1,2,3,4,5}, float32(4), false},
{[]float64{0,1,2,3,4,5}, float64(4), false},
{[]complex64{0,1,2,3,4,5}, complex64(4), false},
{[]complex128{0,1,2,3,4,5}, complex128(4), false},
	{[]bool{true, false, true, false, true, false}, nil, true},
}

func TestDense_Trace(t *testing.T) {
	assert := assert.New(t)
	for i, tts := range traceTests {
		T := New(WithBacking(tts.data), WithShape(2, 3))
		trace, err := T.Trace()

		if checkErr(t, tts.err, err, "Trace", i) {
			continue
		}
		assert.Equal(tts.correct, trace)

		//
		T = New(WithBacking(tts.data))
		_, err = T.Trace()
		if err == nil {
			t.Error("Expected an error when Trace() on non-matrices")
		}
	}
}

var innerTests = []struct {
	a, b           interface{}
	shapeA, shapeB Shape

	correct interface{}
	err     bool
}{
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3, 1}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3, 1}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{1, 3}, float64(5), false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{1, 3}, float64(5), false},

	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3, 1}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3, 1}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{1, 3}, float32(5), false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{1, 3}, float32(5), false},

	// stupids: type differences
	{Range(Int, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float32, 0, 3), Range(Byte, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float64, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, nil, true},
	{Range(Float32, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, nil, true},

	// differing size
	{Range(Float64, 0, 4), Range(Float64, 0, 3), Shape{4}, Shape{3}, nil, true},

	// A is not a matrix
	{Range(Float64, 0, 4), Range(Float64, 0, 3), Shape{2, 2}, Shape{3}, nil, true},
}

func TestDense_Inner(t *testing.T) {
	for i, its := range innerTests {
		a := New(WithShape(its.shapeA...), WithBacking(its.a))
		b := New(WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		if checkErr(t, its.err, err, "Inner", i) {
			continue
		}

		assert.Equal(t, its.correct, T.Data())
	}
}

var matVecMulTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3, 1},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{1, 3},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},

	// Float32s
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3, 1},
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},
	{Range(Float32, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{1, 3},
		Range(Float32, 52, 54), Range(Float32, 100, 102), Shape{2}, Shape{2},
		[]float32{5, 14}, []float32{105, 115}, []float32{110, 129}, Shape{2}, false, false, false},

	// stupids : unpossible shapes (wrong A)
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{6}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad A shape
	{Range(Float64, 0, 8), Range(Float64, 0, 3), Shape{4, 2}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad B shape
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	//stupids: bad reuse
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 55), Range(Float64, 100, 102), Shape{3}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, true},

	//stupids: bad incr shape
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 105), Shape{2}, Shape{5},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},

	// stupids: type mismatch A and B
	{Range(Float64, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float32, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float64, 0, 6), Range(Float32, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{Range(Float32, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B (non-Float)
	{Range(Float64, 0, 6), Range(Int, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch, reuse
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float32, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, true},

	// stupids: type mismatch, incr
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float32, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},

	// stupids: type mismatch, incr not a Number
	{Range(Float64, 0, 6), Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), []bool{true, true, true}, Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},
}

func TestDense_MatVecMul(t *testing.T) {
	assert := assert.New(t)
	for i, mvmt := range matVecMulTests {
		a := New(WithBacking(mvmt.a), WithShape(mvmt.shapeA...))
		b := New(WithBacking(mvmt.b), WithShape(mvmt.shapeB...))

		T, err := a.MatVecMul(b)
		if checkErr(t, mvmt.err, err, "Safe", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correct, T.Data())

		// incr
		incr := New(WithBacking(mvmt.incr), WithShape(mvmt.shapeI...))
		T, err = a.MatVecMul(b, WithIncr(incr))
		if checkErr(t, mvmt.errIncr, err, "WithIncr", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(mvmt.reuse), WithShape(mvmt.shapeR...))
		T, err = a.MatVecMul(b, WithReuse(reuse))

		if checkErr(t, mvmt.errReuse, err, "WithReuse", i) {
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatVecMul(b, WithIncr(incr), WithReuse(reuse))
		if checkErr(t, mvmt.err, err, "WithReuse and WithIncr", i) {
			continue
		}
		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correctIncrReuse, T.Data())
	}
}

var matMulTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, false, false, false},

	// Float32s
	{Range(Float32, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float32, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float32{10, 13, 28, 40}, []float32{110, 114, 130, 143}, []float32{120, 127, 158, 183}, Shape{2, 2}, false, false, false},

	// stupids - bad shape (not matrices):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{6},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids - bad shape (incompatible shapes):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{6, 1},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids - bad shape (bad reuse shape):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 57), Range(Float64, 100, 104), Shape{5}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, false, false, true},

	// stupids - bad shape (bad incr shape):
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{4},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, false, true, false},

	// stupids - type mismatch (a,b)
	{Range(Float64, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids - type mismatch (a,b)
	{Range(Float32, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids type mismatch (b not float)
	{Range(Float64, 0, 6), Range(Int, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids type mismatch (a not float)
	{Range(Int, 0, 6), Range(Int, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, true, false, false},

	// stupids: type mismatch (incr)
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, false, true, false},

	// stupids: type mismatch (reuse)
	{Range(Float64, 0, 6), Range(Float64, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float32, 52, 56), Range(Float64, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float64{10, 13, 28, 40}, []float64{110, 114, 130, 143}, []float64{120, 127, 158, 183}, Shape{2, 2}, false, false, true},

	// stupids: type mismatch (reuse)
	{Range(Float32, 0, 6), Range(Float32, 0, 6), Shape{2, 3}, Shape{3, 2},
		Range(Float64, 52, 56), Range(Float32, 100, 104), Shape{2, 2}, Shape{2, 2},
		[]float32{10, 13, 28, 40}, []float32{110, 114, 130, 143}, []float32{120, 127, 158, 183}, Shape{2, 2}, false, false, true},
}

func TestDense_MatMul(t *testing.T) {
	assert := assert.New(t)
	for i, mmt := range matMulTests {
		a := New(WithBacking(mmt.a), WithShape(mmt.shapeA...))
		b := New(WithBacking(mmt.b), WithShape(mmt.shapeB...))

		T, err := a.MatMul(b)
		if checkErr(t, mmt.err, err, "Safe", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correct, T.Data())

		// incr
		incr := New(WithBacking(mmt.incr), WithShape(mmt.shapeI...))
		T, err = a.MatMul(b, WithIncr(incr))
		if checkErr(t, mmt.errIncr, err, "WithIncr", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(mmt.reuse), WithShape(mmt.shapeR...))
		T, err = a.MatMul(b, WithReuse(reuse))

		if checkErr(t, mmt.errReuse, err, "WithReuse", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatMul(b, WithIncr(incr), WithReuse(reuse))
		if checkErr(t, mmt.err, err, "WithIncr and WithReuse", i) {
			continue
		}
		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correctIncrReuse, T.Data())
	}
}

var outerTests = []linalgTest{
	// Float64s
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		false, false, false},

	// Float32s
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3},
		Range(Float32, 52, 61), Range(Float32, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float32{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float32{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float32{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		false, false, false},

	// stupids - a or b not vector
	{Range(Float64, 0, 3), Range(Float64, 0, 6), Shape{3}, Shape{3, 2},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		true, false, false},

	//	stupids - bad incr shape
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 106), Shape{3, 3}, Shape{3, 2},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		false, true, false},

	// stupids - bad reuse shape
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 58), Range(Float64, 100, 109), Shape{3, 2}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		false, false, true},

	// stupids - b not Float
	{Range(Float64, 0, 3), Range(Int, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		true, false, false},

	// stupids - a not Float
	{Range(Int, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		true, false, false},

	// stupids - a-b type mismatch
	{Range(Float64, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		true, false, false},

	// stupids a-b type mismatch
	{Range(Float32, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3},
		Range(Float64, 52, 61), Range(Float64, 100, 109), Shape{3, 3}, Shape{3, 3},
		[]float64{0, 0, 0, 0, 1, 2, 0, 2, 4}, []float64{100, 101, 102, 103, 105, 107, 106, 109, 112}, []float64{100, 101, 102, 103, 106, 109, 106, 111, 116}, Shape{3, 3},
		true, false, false},
}

func TestDense_Outer(t *testing.T) {
	assert := assert.New(t)
	for i, ot := range outerTests {
		a := New(WithBacking(ot.a), WithShape(ot.shapeA...))
		b := New(WithBacking(ot.b), WithShape(ot.shapeB...))

		T, err := a.Outer(b)
		if checkErr(t, ot.err, err, "Safe", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())

		// incr
		incr := New(WithBacking(ot.incr), WithShape(ot.shapeI...))
		T, err = a.Outer(b, WithIncr(incr))
		if checkErr(t, ot.errIncr, err, "WithIncr", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(ot.reuse), WithShape(ot.shapeR...))
		T, err = a.Outer(b, WithReuse(reuse))
		if checkErr(t, ot.errReuse, err, "WithReuse", i) {
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())

		// reuse AND incr
		T, err = a.Outer(b, WithIncr(incr), WithReuse(reuse))
		if err != nil {
			t.Errorf("Reuse and Incr error'd %+v", err)
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correctIncrReuse, T.Data())
	}
}

var tensorMulTests = []struct {
	a, b           interface{}
	shapeA, shapeB Shape

	reuse, incr    interface{}
	shapeR, shapeI Shape

	correct          interface{}
	correctIncr      interface{}
	correctIncrReuse interface{}
	correctShape     Shape
	err              bool
	errIncr          bool
	errReuse         bool

	axesA, axesB []int
}{
	{a: Range(Float64, 0, 60), b: Range(Float64, 0, 24), shapeA: Shape{3, 4, 5}, shapeB: Shape{4, 3, 2},
		axesA: []int{1, 0}, axesB: []int{0, 1},
		correct: []float64{4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306}, correctShape: Shape{5, 2}},
}

func TestDense_TensorMul(t *testing.T) {
	assert := assert.New(t)
	for i, tmt := range tensorMulTests {
		a := New(WithShape(tmt.shapeA...), WithBacking(tmt.a))
		b := New(WithShape(tmt.shapeB...), WithBacking(tmt.b))

		T, err := a.TensorMul(b, tmt.axesA, tmt.axesB)
		if checkErr(t, tmt.err, err, "Safe", i) {
			continue
		}
		assert.True(tmt.correctShape.Eq(T.Shape()))
		assert.Equal(tmt.correct, T.Data())
	}
}

func TestDot(t *testing.T) {
	assert := assert.New(t)
	var a, b, c, r Tensor
	var A, B, R, R2 Tensor
	var s, s2 Tensor
	var incr Tensor
	var err error
	var expectedShape Shape
	var expectedData []float64
	var expectedScalar float64

	// vector-vector
	t.Log("Vec⋅Vec")
	a = New(Of(Float64), WithShape(3, 1), WithBacking(Range(Float64, 0, 3)))
	b = New(Of(Float64), WithShape(3, 1), WithBacking(Range(Float64, 0, 3)))
	r, err = Dot(a, b)
	expectedShape = Shape{1}
	expectedScalar = float64(5)
	assert.Nil(err)
	assert.Equal(expectedScalar, r.Data())
	assert.True(ScalarShape().Eq(r.Shape()))

	// vector-mat (which is the same as matᵀ*vec)
	t.Log("Vec⋅Mat dot, should be equal to Aᵀb")
	A = New(Of(Float64), WithShape(3, 2), WithBacking(Range(Float64, 0, 6)))
	R, err = Dot(b, A)
	expectedShape = Shape{2}
	expectedData = []float64{10, 13}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// mat-mat
	t.Log("Mat⋅Mat")
	A = New(Of(Float64), WithShape(4, 5), WithBacking(Range(Float64, 0, 20)))
	B = New(Of(Float64), WithShape(5, 10), WithBacking(Range(Float64, 2, 52)))
	R, err = Dot(A, B)
	expectedShape = Shape{4, 10}
	expectedData = []float64{
		320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 870,
		905, 940, 975, 1010, 1045, 1080, 1115, 1150, 1185, 1420, 1480,
		1540, 1600, 1660, 1720, 1780, 1840, 1900, 1960, 1970, 2055, 2140,
		2225, 2310, 2395, 2480, 2565, 2650, 2735,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// T-T
	t.Log("3T⋅3T")
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	B = New(Of(Float64), WithShape(3, 4, 2), WithBacking(Range(Float64, 0, 24)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 3, 3, 2}
	expectedData = []float64{
		28, 34,
		76, 82,
		124, 130,
		76, 98,
		252, 274,
		428, 450,
		124, 162,
		428, 466,
		732, 770,
		//
		172, 226,
		604, 658,
		1036, 1090,
		220, 290,
		780, 850,
		1340, 1410,
		268, 354,
		956, 1042,
		1644, 1730,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// T-T
	t.Log("3T⋅4T")
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	B = New(Of(Float64), WithShape(2, 3, 4, 5), WithBacking(Range(Float64, 0, 120)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 3, 2, 3, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94, 190, 196, 202, 208, 214, 310,
		316, 322, 328, 334, 430, 436, 442, 448, 454, 550, 556,
		562, 568, 574, 670, 676, 682, 688, 694, 190, 212, 234,
		256, 278, 630, 652, 674, 696, 718, 1070, 1092, 1114, 1136,
		1158, 1510, 1532, 1554, 1576, 1598, 1950, 1972, 1994, 2016, 2038,
		2390, 2412, 2434, 2456, 2478, 310, 348, 386, 424, 462, 1070,
		1108, 1146, 1184, 1222, 1830, 1868, 1906, 1944, 1982, 2590, 2628,
		2666, 2704, 2742, 3350, 3388, 3426, 3464, 3502, 4110, 4148, 4186,
		4224, 4262, 430, 484, 538, 592, 646, 1510, 1564, 1618, 1672,
		1726, 2590, 2644, 2698, 2752, 2806, 3670, 3724, 3778, 3832, 3886,
		4750, 4804, 4858, 4912, 4966, 5830, 5884, 5938, 5992, 6046, 550,
		620, 690, 760, 830, 1950, 2020, 2090, 2160, 2230, 3350, 3420,
		3490, 3560, 3630, 4750, 4820, 4890, 4960, 5030, 6150, 6220, 6290,
		6360, 6430, 7550, 7620, 7690, 7760, 7830, 670, 756, 842, 928,
		1014, 2390, 2476, 2562, 2648, 2734, 4110, 4196, 4282, 4368, 4454,
		5830, 5916, 6002, 6088, 6174, 7550, 7636, 7722, 7808, 7894, 9270,
		9356, 9442, 9528, 9614,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// T-v
	t.Log("3T⋅Vec")
	b = New(Of(Float64), WithShape(4), WithBacking(Range(Float64, 0, 4)))
	R, err = Dot(A, b)
	expectedShape = Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// v-T
	t.Log("Vec⋅3T")
	R2, err = Dot(b, B)
	expectedShape = Shape{2, 3, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94,
		190, 196, 202, 208, 214,
		310, 316, 322, 328, 334,
		430, 436, 442, 448, 454,
		550, 556, 562, 568, 574,
		670, 676, 682, 688, 694,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R2.Data())
	assert.Equal(expectedShape, R2.Shape())
	// m-3T
	t.Log("Mat⋅3T")
	A = New(Of(Float64), WithShape(2, 4), WithBacking(Range(Float64, 0, 8)))
	B = New(Of(Float64), WithShape(2, 4, 5), WithBacking(Range(Float64, 0, 40)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 2, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94,
		190, 196, 202, 208, 214,
		190, 212, 234, 256, 278,
		630, 652, 674, 696, 718,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// test reuse
	// m-v with reuse
	t.Log("Mat⋅Vec with reuse")
	R = New(Of(Float64), WithShape(2))
	R2, err = Dot(A, b, WithReuse(R))
	expectedShape = Shape{2}
	expectedData = []float64{14, 38}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// 3T-vec with reuse
	t.Logf("3T⋅vec with reuse")
	R = New(Of(Float64), WithShape(6))
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	R2, err = Dot(A, b, WithReuse(R))
	expectedShape = Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R2.Data())
	assert.Equal(expectedShape, R2.Shape())
	// v-m
	t.Log("vec⋅Mat with reuse")
	R = New(Of(Float64), WithShape(2))
	a = New(Of(Float64), WithShape(4), WithBacking(Range(Float64, 0, 4)))
	B = New(Of(Float64), WithShape(4, 2), WithBacking(Range(Float64, 0, 8)))
	R2, err = Dot(a, B, WithReuse(R))
	expectedShape = Shape{2}
	expectedData = []float64{28, 34}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())
	// test incr
	incrBack := make([]float64, 2)
	copy(incrBack, expectedData)
	incr = New(Of(Float64), WithBacking(incrBack), WithShape(2))
	R, err = Dot(a, B, WithIncr(incr))
	vecf64.Scale(expectedData, 2)
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// The Nearly Stupids
	s = New(FromScalar(5.0))
	s2 = New(FromScalar(10.0))
	R, err = Dot(s, s2)
	assert.Nil(err)
	assert.True(R.IsScalar())
	assert.Equal(float64(50), R.Data())
	R.Zero()
	R2, err = Dot(s, s2, WithReuse(R))
	assert.Nil(err)
	assert.True(R2.IsScalar())
	assert.Equal(float64(50), R2.Data())

	R, err = Dot(s, A)
	expectedData = vecf64.Range(0, 24)
	vecf64.Scale(expectedData, 5)
	assert.Nil(err)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())
	R.Zero()
	R2, err = Dot(s, A, WithReuse(R))
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(A.Shape(), R2.Shape())
	assert.Equal(expectedData, R2.Data())
	R, err = Dot(A, s)
	assert.Nil(err)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())
	R.Zero()
	R2, err = Dot(A, s, WithReuse(R))
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(A.Shape(), R2.Shape())
	assert.Equal(expectedData, R2.Data())
	incr = New(Of(Float64), WithShape(R2.Shape()...))
	copy(incr.Data().([]float64), expectedData)
	incr2 := incr.Clone().(*Dense) // backup a copy for the following test
	vecf64.Scale(expectedData, 2)
	R, err = Dot(A, s, WithIncr(incr))
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())
	incr = incr2

	R, err = Dot(s, A, WithIncr(incr))
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())
	incr = New(Of(Float64), FromScalar(float64(50)))

	R, err = Dot(s, s2, WithIncr(incr))
	assert.Nil(err)
	assert.Equal(R, incr)
	assert.True(R.IsScalar())
	assert.Equal(float64(100), R.Data())

	/* HERE BE STUPIDS */
	// different sizes of vectors
	c = New(Of(Float64), WithShape(1, 100))
	_, err = Dot(a, c)
	assert.NotNil(err)
	// vector mat, but with shape mismatch
	B = New(Of(Float64), WithShape(2, 3), WithBacking(Range(Float64, 0, 6)))
	_, err = Dot(b, B)
	assert.NotNil(err)
	// mat-mat but wrong reuse size
	A = New(Of(Float64), WithShape(2, 2))
	R = New(Of(Float64), WithShape(5, 10))
	_, err = Dot(A, B, WithReuse(R))
	assert.NotNil(err)
	// mat-vec but wrong reuse size
	b = New(Of(Float64), WithShape(2))
	_, err = Dot(A, b, WithReuse(R))
	assert.NotNil(err)
	// T-T but misaligned shape
	A = New(Of(Float64), WithShape(2, 3, 4))
	B = New(Of(Float64), WithShape(4, 2, 3))
	_, err = Dot(A, B)
	assert.NotNil(err)
}

func TestOneDot(t *testing.T) {
	assert := assert.New(t)
	A := New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	b := New(Of(Float64), WithShape(4), WithBacking(Range(Float64, 0, 4)))

	R, err := Dot(A, b)
	expectedShape := Shape{2, 3}
	expectedData := []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// 3T-vec with reuse
	t.Logf("3T⋅vec with reuse")
	R.Zero()
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	R2, err := Dot(A, b, WithReuse(R))
	expectedShape = Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R2.Data())
	assert.Equal(expectedShape, R2.Shape())
}
