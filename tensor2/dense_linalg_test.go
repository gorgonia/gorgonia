package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type linalgTest struct {
	a, b           Array
	shapeA, shapeB Shape

	reuse, incr    Array
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
	data Array

	correct interface{}
	err     bool
}{
	{f64s{0, 1, 2, 3, 4, 5}, float64(4), false},
	{f32s{0, 1, 2, 3, 4, 5}, float32(4), false},
	{ints{0, 1, 2, 3, 4, 5}, int(4), false},
	{i64s{0, 1, 2, 3, 4, 5}, int64(4), false},
	{i32s{0, 1, 2, 3, 4, 5}, int32(4), false},
	{u8s{0, 1, 2, 3, 4, 5}, byte(4), false},

	// compatible types
	{f64sDummy{0, 1, 2, 3, 4, 5}, float64(4), false},

	{bs{true, false, true, false, true, false}, nil, true},
}

func TestDense_Trace(t *testing.T) {
	assert := assert.New(t)
	for _, tts := range traceTests {
		T := New(WithBacking(tts.data), WithShape(2, 3))
		trace, err := T.Trace()

		switch {
		case tts.err:
			if err == nil {
				t.Errorf("Expected an error: %v", T.Data())
			}
			continue
		case !tts.err && err != nil:
			t.Errorf("%+v", err)
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
	a, b           Array
	shapeA, shapeB Shape

	correct interface{}
	err     bool
}{
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3}, Shape{3}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{3, 1}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{3, 1}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{3, 1}, Shape{1, 3}, []float64{5}, false},
	{Range(Float64, 0, 3), Range(Float64, 0, 3), Shape{1, 3}, Shape{1, 3}, []float64{5}, false},

	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3}, Shape{3}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{3, 1}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{3, 1}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{3, 1}, Shape{1, 3}, []float32{5}, false},
	{Range(Float32, 0, 3), Range(Float32, 0, 3), Shape{1, 3}, Shape{1, 3}, []float32{5}, false},

	// compatible
	{f64sDummy{0, 1, 2}, f64sDummy{0, 1, 2}, Shape{3}, Shape{3}, []float64{5}, false},
	{f32sDummy{0, 1, 2}, f32sDummy{0, 1, 2}, Shape{3}, Shape{3}, []float32{5}, false},

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
	for _, its := range innerTests {
		a := New(WithShape(its.shapeA...), WithBacking(its.a))
		b := New(WithShape(its.shapeB...), WithBacking(its.b))

		T, err := a.Inner(b)
		switch {
		case its.err:
			if err == nil {
				t.Error("Expected an error!")
			}
			continue
		case !its.err && err != nil:
			t.Errorf("%+v", err)
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

	// compatibles
	{f64sDummy{0, 1, 2, 3, 4, 5}, Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 102), Shape{2}, Shape{2},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, false, false},
	{f32sDummy{0, 1, 2, 3, 4, 5}, Range(Float32, 0, 3), Shape{2, 3}, Shape{3},
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
	{f64sDummy{0, 1, 2, 3, 4, 5}, Range(Float32, 0, 3), Shape{2, 3}, Shape{3},
		Range(Float64, 52, 54), Range(Float64, 100, 103), Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, true, false, false},

	// stupids: type mismatch A and B
	{f32sDummy{0, 1, 2, 3, 4, 5}, Range(Float64, 0, 3), Shape{2, 3}, Shape{3},
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
		Range(Float64, 52, 54), bs{true, true, true}, Shape{2}, Shape{3},
		[]float64{5, 14}, []float64{105, 115}, []float64{110, 129}, Shape{2}, false, true, false},
}

func TestDense_MatVecMul(t *testing.T) {
	assert := assert.New(t)
	for i, mvmt := range matVecMulTests {
		a := New(WithBacking(mvmt.a), WithShape(mvmt.shapeA...))
		b := New(WithBacking(mvmt.b), WithShape(mvmt.shapeB...))

		T, err := a.MatVecMul(b)
		switch {
		case mvmt.err:
			if err == nil {
				t.Errorf("Safe Test (%d) : Expected an error | a: %v(%T) | b %v(%T)", i, mvmt.a, mvmt.a, mvmt.b, mvmt.b)
			}
			continue
		case !mvmt.err && err != nil:
			t.Errorf("Safe Test(%d) error'd: %+v", i, err)
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correct, T.Data())

		// incr
		incr := New(WithBacking(mvmt.incr), WithShape(mvmt.shapeI...))
		T, err = a.MatVecMul(b, WithIncr(incr))
		switch {
		case mvmt.errIncr:
			if err == nil {
				t.Errorf("WithIncr Test (%d): Expected an error", i)
			}
			continue
		case !mvmt.errIncr && err != nil:
			t.Errorf("WithIncr Test (%d) err: %+v", i, err)
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(mvmt.reuse), WithShape(mvmt.shapeR...))
		T, err = a.MatVecMul(b, WithReuse(reuse))

		switch {
		case mvmt.errReuse:
			if err == nil {
				t.Error("Expected an error withReuse")
			}
			continue
		case !mvmt.errReuse && err != nil:
			t.Error("WithReuse (%d) err: %+v", i, err)
			continue
		}

		assert.True(mvmt.correctShape.Eq(T.Shape()))
		assert.Equal(mvmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatVecMul(b, WithIncr(incr), WithReuse(reuse))
		if err != nil {
			t.Errorf("Reuse and Incr error'd %+v", err)
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

		switch {
		case mmt.err:
			if err == nil {
				t.Errorf("Safe Test (%d) : Expected an error | a: %v(%T) | b %v(%T)", i, mmt.a, mmt.a, mmt.b, mmt.b)
			}
			continue
		case !mmt.err && err != nil:
			t.Errorf("Safe Test(%d) error'd: %+v", i, err)
			continue
		}

		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correct, T.Data())

		// incr
		incr := New(WithBacking(mmt.incr), WithShape(mmt.shapeI...))
		T, err = a.MatMul(b, WithIncr(incr))
		switch {
		case mmt.errIncr:
			if err == nil {
				t.Errorf("WithIncr Test (%d): Expected an error", i)
			}
			continue
		case !mmt.errIncr && err != nil:
			t.Errorf("WithIncr Test (%d) err: %+v", i, err)
			continue
		}

		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(mmt.reuse), WithShape(mmt.shapeR...))
		T, err = a.MatMul(b, WithReuse(reuse))

		switch {
		case mmt.errReuse:
			if err == nil {
				t.Error("Expected an error withReuse")
			}
			continue
		case !mmt.errReuse && err != nil:
			t.Error("WithReuse (%d) err: %+v", i, err)
			continue
		}

		assert.True(mmt.correctShape.Eq(T.Shape()))
		assert.Equal(mmt.correct, T.Data())

		// reuse AND incr
		T, err = a.MatMul(b, WithIncr(incr), WithReuse(reuse))
		if err != nil {
			t.Errorf("Reuse and Incr error'd %+v", err)
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
		switch {
		case ot.err:
			if err == nil {
				t.Errorf("Safe Test(%d): Expected an error", i)
			}
			continue
		case !ot.err && err != nil:
			t.Errorf("Safe Test(%d) errored: %+v", i, err)
			continue
		}
		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correct, T.Data())

		// incr
		incr := New(WithBacking(ot.incr), WithShape(ot.shapeI...))
		T, err = a.Outer(b, WithIncr(incr))
		switch {
		case ot.errIncr:
			if err == nil {
				t.Errorf("WithIncr Test (%d): Expected an error", i)
			}
			continue
		case !ot.errIncr && err != nil:
			t.Errorf("WithIncr Test (%d) err: %+v", i, err)
			continue
		}

		assert.True(ot.correctShape.Eq(T.Shape()))
		assert.Equal(ot.correctIncr, T.Data())

		// reuse
		reuse := New(WithBacking(ot.reuse), WithShape(ot.shapeR...))
		T, err = a.Outer(b, WithReuse(reuse))

		switch {
		case ot.errReuse:
			if err == nil {
				t.Error("Expected an error withReuse")
			}
			continue
		case !ot.errReuse && err != nil:
			t.Error("WithReuse (%d) err: %+v", i, err)
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
	a, b           Array
	shapeA, shapeB Shape

	reuse, incr    Array
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
		switch {
		case tmt.err:
			if err == nil {
				t.Errorf("Safe Test (%d) : Expected an error ", i)
			}
			continue
		case !tmt.err && err != nil:
			t.Errorf("Safe Test(%d) error'd: %+v", i, err)
			continue
		}

		assert.True(tmt.correctShape.Eq(T.Shape()))
		assert.Equal(tmt.correct, T.Data())
	}
}
