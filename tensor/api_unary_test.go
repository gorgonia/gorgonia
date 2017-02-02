package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var pointwiseSquareTests = []struct {
	a     interface{}
	reuse interface{}
	incr  interface{}

	correct     interface{}
	correctIncr interface{}
	err         bool
	errReuse    bool
}{
	{[]float64{0, 1, 2, -3, 4}, []float64{0, 10, 20, 30, 40}, []float64{0, 100, 200, 300, 400}, []float64{0, 1, 4, 9, 16}, []float64{0, 101, 204, 309, 416}, false, false},
	{[]float32{0, 1, 2, -3, 4}, []float32{0, 10, 20, 30, 40}, []float32{0, 100, 200, 300, 400}, []float32{0, 1, 4, 9, 16}, []float32{0, 101, 204, 309, 416}, false, false},
	{[]int{0, 1, 2, -3, 4}, []int{0, 10, 20, 30, 40}, []int{0, 100, 200, 300, 400}, []int{0, 1, 4, 9, 16}, []int{0, 101, 204, 309, 416}, false, false},

	// stupids: non-number
	{[]bool{true, false}, nil, nil, nil, nil, true, false},

	// stupids: wrong reuse type
	{[]float64{2, 4}, []float32{0, 0}, nil, []float64{4, 16}, nil, false, true},

	// stupids: wrong reuse size
	{[]float64{2, 4, 6}, []float64{1}, nil, []float64{4, 16, 36}, nil, false, true},
}

func TestPointwiseSquare(t *testing.T) {
	assert := assert.New(t)

	for i, pst := range pointwiseSquareTests {
		var a, reuse, incr, T Tensor
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
		assert.Equal(pst.correct, reuse.Data()) // ensure that the reuse has been clobbered

		// incr
		a = New(WithBacking(pst.a))
		incr = New(WithBacking(pst.incr))
		T, err = PointwiseSquare(a, WithIncr(incr))
		if checkErr(t, pst.err, err, "Incr", i) {
			continue
		}
		assert.Equal(pst.correctIncr, T.Data())
		assert.Equal(pst.correctIncr, pst.incr)

		// unsafe
		a = New(WithBacking(pst.a))
		T, err = PointwiseSquare(a, UseUnsafe())

		if checkErr(t, pst.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(pst.correct, T.Data())
		assert.Equal(pst.correct, pst.a) // ensures `a` has been clobbered
	}
}

var sqrtTests = []struct {
	a, reuse, incr interface{}

	correct     interface{}
	correctIncr interface{}
	err         bool
	errReuse    bool
}{
	{[]float64{0, 1, 4, 9, 16}, []float64{0, 10, 20, 30, 40}, []float64{0, 100, 200, 300, 400}, []float64{0, 1, 2, 3, 4}, []float64{0, 101, 202, 303, 404}, false, false},
	{[]float32{0, 1, 4, 9, 16}, []float32{0, 10, 20, 30, 40}, []float32{0, 100, 200, 300, 400}, []float32{0, 1, 2, 3, 4}, []float32{0, 101, 202, 303, 404}, false, false},

	// unsupported for now
	{[]int{0, 1, 4, 9, 16}, []int{0, 10, 20, 30, 40}, []int{0, 100, 200, 300, 400}, []int{0, 1, 2, 3, 4}, []int{0, 101, 202, 303, 404}, true, true},

	// stupids: wrong resize shape
	{[]float32{0, 1, 4, 9, 16}, []float32{0, 10}, nil, []float32{0, 1, 2, 3, 4}, nil, false, true},
}

func TestSqrt(t *testing.T) {
	assert := assert.New(t)

	for i, st := range sqrtTests {
		var a, reuse, incr, T Tensor
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
		assert.Equal(st.correct, st.reuse) // ensure that the reuse has been clobbered

		// incr
		a = New(WithBacking(st.a))
		incr = New(WithBacking(st.incr))
		T, err = Sqrt(a, WithIncr(incr))
		if checkErr(t, st.err, err, "incr", i) {
			continue
		}
		assert.Equal(st.correctIncr, T.Data())
		assert.Equal(st.correctIncr, st.incr) // ensure that the incr has been clobbered

		// unsafe
		a = New(WithBacking(st.a))
		T, err = Sqrt(a, UseUnsafe())

		if checkErr(t, st.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(st.correct, T.Data())
		assert.Equal(st.correct, st.a) // ensure a has been clobbered
	}
}

var invSqrtTests = []struct {
	a, reuse, incr interface{}

	correct     interface{}
	correctIncr interface{}
	err         bool
	errReuse    bool
}{
	{[]float64{1, 4, 16}, []float64{10, 20, 40}, []float64{100, 200, 400}, []float64{1, 0.5, 0.25}, []float64{101, 200.5, 400.25}, false, false},
	{[]float32{1, 4, 16}, []float32{10, 20, 40}, []float32{100, 200, 400}, []float32{1, 0.5, 0.25}, []float32{101, 200.5, 400.25}, false, false},

	// unsupported for now
	{[]int{1, 4, 16}, []int{10, 20, 40}, nil, []int{0, 0, 0}, nil, true, true},

	// stupids: wrong resize shape
	{[]float32{1, 4, 16}, []float32{0, 10}, nil, []float32{1, 0.5, 0.25}, nil, false, true},
}

func TestInvSqrt(t *testing.T) {
	assert := assert.New(t)

	for i, ist := range invSqrtTests {
		var a, reuse, T, incr Tensor
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
		assert.Equal(ist.correct, ist.reuse) // ensure that the reuse has been clobbered

		//incr
		a = New(WithBacking(ist.a))
		incr = New(WithBacking(ist.incr))
		T, err = InvSqrt(a, WithIncr(incr))
		if checkErr(t, ist.err, err, "Incr", i) {
			continue
		}
		assert.Equal(ist.correctIncr, T.Data())
		assert.Equal(ist.correctIncr, ist.incr) // esnure that the incr array has been clobbered

		// unsafe
		a = New(WithBacking(ist.a))
		T, err = InvSqrt(a, UseUnsafe())

		if checkErr(t, ist.err, err, "Unsafe", i) {
			continue
		}
		assert.Equal(ist.correct, T.Data())
		assert.Equal(ist.correct, ist.a) // ensure a has been clobbered
	}
}
