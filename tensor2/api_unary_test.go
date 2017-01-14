package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var pointwiseSquareTests = []struct {
	a     Array
	reuse Array

	correct  interface{}
	err      bool
	errReuse bool
}{
	{f64s{0, 1, 2, -3, 4}, f64s{0, 10, 20, 30, 40}, []float64{0, 1, 4, 9, 16}, false, false},
	{f32s{0, 1, 2, -3, 4}, f32s{0, 10, 20, 30, 40}, []float32{0, 1, 4, 9, 16}, false, false},
	{ints{0, 1, 2, -3, 4}, ints{0, 10, 20, 30, 40}, []int{0, 1, 4, 9, 16}, false, false},
	{i64s{0, 1, 2, -3, 4}, i64s{0, 10, 20, 30, 40}, []int64{0, 1, 4, 9, 16}, false, false},
	{i32s{0, 1, 2, -3, 4}, i32s{0, 10, 20, 30, 40}, []int32{0, 1, 4, 9, 16}, false, false},
	{u8s{0, 1, 2, 3, 4}, u8s{0, 10, 20, 30, 40}, []byte{0, 1, 4, 9, 16}, false, false},

	// stupids: non-number
	{bs{true, false}, nil, nil, true, false},

	// stupids: wrong reuse type
	{f64s{2, 4}, f32s{0, 0}, []float64{4, 16}, false, true},

	// stupids: wrong reuse size
	{f64s{2, 4, 6}, f64s{1}, []float64{4, 16, 36}, false, true},
}

func TestPointwiseSquare(t *testing.T) {
	assert := assert.New(t)

	for i, pst := range pointwiseSquareTests {
		var a, reuse, T Tensor
		var err error
		a = New(WithBacking(pst.a))
		T, err = PointwiseSquare(a)

		if checkErr(t, pst.err, err, "Safe", i) {
			continue
		}
		assert.Equal(pst.correct, T.Data())

		// reuse
		a = New(WithBacking(pst.a))
		reuse = New(WithBacking(pst.reuse))
		T, err = PointwiseSquare(a, WithReuse(reuse))

		if checkErr(t, pst.errReuse, err, "Reuse", i) {
			continue
		}
		assert.Equal(pst.correct, T.Data())
		assert.Equal(pst.correct, pst.reuse.Data()) // ensure that the reuse has been clobbered

		// unsafe
		a = New(WithBacking(pst.a))
		T, err = PointwiseSquare(a, UseUnsafe())

		if checkErr(t, pst.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(pst.correct, T.Data())
		assert.Equal(pst.correct, pst.a.Data()) // ensure a has been clobbered
	}
}

var sqrtTests = []struct {
	a, reuse Array

	correct  interface{}
	err      bool
	errReuse bool
}{
	{f64s{0, 1, 4, 9, 16}, f64s{0, 10, 20, 30, 40}, []float64{0, 1, 2, 3, 4}, false, false},
	{f32s{0, 1, 4, 9, 16}, f32s{0, 10, 20, 30, 40}, []float32{0, 1, 2, 3, 4}, false, false},

	// unsupported for now
	{ints{0, 1, 4, 9, 16}, ints{0, 10, 20, 30, 40}, []int{0, 1, 2, 3, 4}, true, true},
	{i64s{0, 1, 4, 9, 16}, i64s{0, 10, 20, 30, 40}, []int64{0, 1, 2, 3, 4}, true, true},
	{i32s{0, 1, 4, 9, 16}, i32s{0, 10, 20, 30, 40}, []int32{0, 1, 2, 3, 4}, true, true},
	{u8s{0, 1, 4, 9, 16}, u8s{0, 10, 20, 30, 40}, []byte{0, 1, 2, 3, 4}, true, true},

	// stupids: wrong resize shape
	{f32s{0, 1, 4, 9, 16}, f32s{0, 10}, []float32{0, 1, 2, 3, 4}, false, true},
}

func TestSqrt(t *testing.T) {
	assert := assert.New(t)

	for i, st := range sqrtTests {
		var a, reuse, T Tensor
		var err error
		a = New(WithBacking(st.a))
		T, err = Sqrt(a)

		if checkErr(t, st.err, err, "Safe", i) {
			continue
		}
		assert.Equal(st.correct, T.Data())

		// reuse
		a = New(WithBacking(st.a))
		reuse = New(WithBacking(st.reuse))
		T, err = Sqrt(a, WithReuse(reuse))

		if checkErr(t, st.errReuse, err, "Reuse", i) {
			continue
		}
		assert.Equal(st.correct, T.Data())
		assert.Equal(st.correct, st.reuse.Data()) // ensure that the reuse has been clobbered

		// unsafe
		a = New(WithBacking(st.a))
		T, err = Sqrt(a, UseUnsafe())

		if checkErr(t, st.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(st.correct, T.Data())
		assert.Equal(st.correct, st.a.Data()) // ensure a has been clobbered
	}
}

var invSqrtTests = []struct {
	a, reuse Array

	correct  interface{}
	err      bool
	errReuse bool
}{
	{f64s{1, 4, 16}, f64s{10, 20, 40}, []float64{1, 0.5, 0.25}, false, false},
	{f32s{1, 4, 16}, f32s{10, 20, 40}, []float32{1, 0.5, 0.25}, false, false},

	// unsupported for now
	{ints{1, 4, 16}, ints{10, 20, 40}, []int{0, 0, 0}, true, true},
	{i64s{1, 4, 16}, i64s{10, 20, 40}, []int64{0, 0, 0}, true, true},
	{i32s{1, 4, 16}, i32s{10, 20, 40}, []int32{0, 0, 0}, true, true},
	{u8s{1, 4, 16}, u8s{10, 20, 40}, []byte{0, 0, 0}, true, true},

	// stupids: wrong resize shape
	{f32s{1, 4, 16}, f32s{0, 10}, []float32{1, 0.5, 0.25}, false, true},
}

func TestInvSqrt(t *testing.T) {
	assert := assert.New(t)

	for i, ist := range invSqrtTests {
		var a, reuse, T Tensor
		var err error
		a = New(WithBacking(ist.a))
		T, err = InvSqrt(a)

		if checkErr(t, ist.err, err, "Safe", i) {
			continue
		}
		assert.Equal(ist.correct, T.Data())

		// reuse
		a = New(WithBacking(ist.a))
		reuse = New(WithBacking(ist.reuse))
		T, err = InvSqrt(a, WithReuse(reuse))

		if checkErr(t, ist.errReuse, err, "Reuse", i) {
			continue
		}
		assert.Equal(ist.correct, T.Data())
		assert.Equal(ist.correct, ist.reuse.Data()) // ensure that the reuse has been clobbered

		// unsafe
		a = New(WithBacking(ist.a))
		T, err = InvSqrt(a, UseUnsafe())

		if checkErr(t, ist.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(ist.correct, T.Data())
		assert.Equal(ist.correct, ist.a.Data()) // ensure a has been clobbered
	}
}

var clampTests = []struct {
	a, reuse Array
	min, max interface{}

	correct       interface{}
	err, errReuse bool
}{
	{f64s{-100, -10, -5, 0, 5, 10, 100}, f64s{-52, -50, -52, 52, 53, 52, 52}, float64(-10), float64(10), []float64{-10, -10, -5, 0, 5, 10, 10}, false, false},
	{f32s{-100, -10, -5, 0, 5, 10, 100}, f32s{-52, -50, -52, 52, 53, 52, 52}, float32(-10), float32(10), []float32{-10, -10, -5, 0, 5, 10, 10}, false, false},
	{ints{-100, -10, -5, 0, 5, 10, 100}, ints{-52, -50, -52, 52, 53, 52, 52}, int(-10), int(10), []int{-10, -10, -5, 0, 5, 10, 10}, false, false},
	{i64s{-100, -10, -5, 0, 5, 10, 100}, i64s{-52, -50, -52, 52, 53, 52, 52}, int64(-10), int64(10), []int64{-10, -10, -5, 0, 5, 10, 10}, false, false},
	{i32s{-100, -10, -5, 0, 5, 10, 100}, i32s{-52, -50, -52, 52, 53, 52, 52}, int32(-10), int32(10), []int32{-10, -10, -5, 0, 5, 10, 10}, false, false},
	{u8s{0, 5, 10, 100}, u8s{52, 53, 52, 52}, byte(5), byte(10), []byte{5, 5, 10, 10}, false, false},

	// stupids: unsupported Dtypes
	{bs{true, false}, nil, nil, nil, nil, true, true},
	// stupids: wrong reuse size
	{f64s{-100, -10, -5, 0, 5, 10, 100}, f64s{-52, 52, 53, 52, 52}, float64(-10), float64(10), []float64{-10, -10, -5, 0, 5, 10, 10}, false, true},
	// stupids, wrong min type
	{f64s{-100, -10, -5, 0, 5, 10, 100}, nil, float32(-10), float64(10), nil, true, true},
	{f32s{-100, -10, -5, 0, 5, 10, 100}, nil, float64(-10), float32(10), nil, true, true},
	{ints{-100, -10, -5, 0, 5, 10, 100}, nil, float32(-10), int(10), nil, true, true},
	{i64s{-100, -10, -5, 0, 5, 10, 100}, nil, float32(-10), int64(10), nil, true, true},
	{i32s{-100, -10, -5, 0, 5, 10, 100}, nil, float32(-10), int32(10), nil, true, true},
	{u8s{0, 5, 10, 100}, u8s{52, 53, 52, 52}, float32(5), byte(10), nil, true, true},

	// stupids, wrong max type
	{f64s{-100, -10, -5, 0, 5, 10, 100}, nil, float64(-10), float32(10), nil, true, true},
	{f32s{-100, -10, -5, 0, 5, 10, 100}, nil, float32(-10), float64(10), nil, true, true},
	{ints{-100, -10, -5, 0, 5, 10, 100}, nil, int(-10), float64(10), nil, true, true},
	{i64s{-100, -10, -5, 0, 5, 10, 100}, nil, int64(-10), float64(10), nil, true, true},
	{i32s{-100, -10, -5, 0, 5, 10, 100}, nil, int32(-10), float64(10), nil, true, true},
	{u8s{0, 5, 10, 100}, u8s{52, 53, 52, 52}, byte(5), float64(10), nil, true, true},
}

func TestClamp(t *testing.T) {
	assert := assert.New(t)
	for i, ct := range clampTests {
		var T Tensor
		var err error

		a := New(WithBacking(ct.a))
		T, err = Clamp(a, ct.min, ct.max)

		if checkErr(t, ct.err, err, "Safe", i) {
			continue
		}
		assert.Equal(ct.correct, T.Data())

		// swap min and max, because people get confused wrt to the order of things
		a = New(WithBacking(ct.a))
		T, err = Clamp(a, ct.max, ct.min)

		if checkErr(t, ct.err, err, "Safe swapped", i) {
			continue
		}
		assert.Equal(ct.correct, T.Data())

		// swap min and max, because people get confused wrt to the order of things
		a = New(WithBacking(ct.a))
		reuse := New(WithBacking(ct.reuse))
		T, err = Clamp(a, ct.max, ct.min, WithReuse(reuse))

		if checkErr(t, ct.errReuse, err, "Reuse", i) {
			continue
		}
		assert.Equal(ct.correct, T.Data())
		assert.Equal(ct.correct, ct.reuse.Data()) // reuse is clobbered

		// unsafe
		a = New(WithBacking(ct.a))
		T, err = Clamp(a, ct.max, ct.min, UseUnsafe())

		if checkErr(t, ct.errReuse, err, "Unsafe", i) {
			continue
		}
		assert.Equal(ct.correct, T.Data())
		assert.Equal(ct.correct, ct.a.Data()) // a is clobbered
	}
}
