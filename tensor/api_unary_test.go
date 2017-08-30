package tensor

import (
	"math/rand"
	"testing"
	"testing/quick"
	"time"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE BY Genlib V1. DO NOT EDIT
*/

var clampTests = []struct {
	a, reuse      interface{}
	min, max      interface{}
	correct       interface{}
	correctSliced interface{}
}{
	{[]int{1, 2, 3, 4}, []int{10, 20, 30, 40}, int(2), int(3), []int{2, 2, 3, 3}, []int{2, 2, 3}},
	{[]int8{1, 2, 3, 4}, []int8{10, 20, 30, 40}, int8(2), int8(3), []int8{2, 2, 3, 3}, []int8{2, 2, 3}},
	{[]int16{1, 2, 3, 4}, []int16{10, 20, 30, 40}, int16(2), int16(3), []int16{2, 2, 3, 3}, []int16{2, 2, 3}},
	{[]int32{1, 2, 3, 4}, []int32{10, 20, 30, 40}, int32(2), int32(3), []int32{2, 2, 3, 3}, []int32{2, 2, 3}},
	{[]int64{1, 2, 3, 4}, []int64{10, 20, 30, 40}, int64(2), int64(3), []int64{2, 2, 3, 3}, []int64{2, 2, 3}},
	{[]uint{1, 2, 3, 4}, []uint{10, 20, 30, 40}, uint(2), uint(3), []uint{2, 2, 3, 3}, []uint{2, 2, 3}},
	{[]uint8{1, 2, 3, 4}, []uint8{10, 20, 30, 40}, uint8(2), uint8(3), []uint8{2, 2, 3, 3}, []uint8{2, 2, 3}},
	{[]uint16{1, 2, 3, 4}, []uint16{10, 20, 30, 40}, uint16(2), uint16(3), []uint16{2, 2, 3, 3}, []uint16{2, 2, 3}},
	{[]uint32{1, 2, 3, 4}, []uint32{10, 20, 30, 40}, uint32(2), uint32(3), []uint32{2, 2, 3, 3}, []uint32{2, 2, 3}},
	{[]uint64{1, 2, 3, 4}, []uint64{10, 20, 30, 40}, uint64(2), uint64(3), []uint64{2, 2, 3, 3}, []uint64{2, 2, 3}},
	{[]float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, float32(2), float32(3), []float32{2, 2, 3, 3}, []float32{2, 2, 3}},
	{[]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}, float64(2), float64(3), []float64{2, 2, 3, 3}, []float64{2, 2, 3}},
}

var clampTestsMasked = []struct {
	a, reuse      interface{}
	min, max      interface{}
	correct       interface{}
	correctSliced interface{}
}{
	{[]int{1, 2, 3, 4}, []int{1, 20, 30, 40}, int(2), int(3), []int{1, 2, 3, 3}, []int{1, 2, 3}},
	{[]int8{1, 2, 3, 4}, []int8{1, 20, 30, 40}, int8(2), int8(3), []int8{1, 2, 3, 3}, []int8{1, 2, 3}},
	{[]int16{1, 2, 3, 4}, []int16{1, 20, 30, 40}, int16(2), int16(3), []int16{1, 2, 3, 3}, []int16{1, 2, 3}},
	{[]int32{1, 2, 3, 4}, []int32{1, 20, 30, 40}, int32(2), int32(3), []int32{1, 2, 3, 3}, []int32{1, 2, 3}},
	{[]int64{1, 2, 3, 4}, []int64{1, 20, 30, 40}, int64(2), int64(3), []int64{1, 2, 3, 3}, []int64{1, 2, 3}},
	{[]uint{1, 2, 3, 4}, []uint{1, 20, 30, 40}, uint(2), uint(3), []uint{1, 2, 3, 3}, []uint{1, 2, 3}},
	{[]uint8{1, 2, 3, 4}, []uint8{1, 20, 30, 40}, uint8(2), uint8(3), []uint8{1, 2, 3, 3}, []uint8{1, 2, 3}},
	{[]uint16{1, 2, 3, 4}, []uint16{1, 20, 30, 40}, uint16(2), uint16(3), []uint16{1, 2, 3, 3}, []uint16{1, 2, 3}},
	{[]uint32{1, 2, 3, 4}, []uint32{1, 20, 30, 40}, uint32(2), uint32(3), []uint32{1, 2, 3, 3}, []uint32{1, 2, 3}},
	{[]uint64{1, 2, 3, 4}, []uint64{1, 20, 30, 40}, uint64(2), uint64(3), []uint64{1, 2, 3, 3}, []uint64{1, 2, 3}},
	{[]float32{1, 2, 3, 4}, []float32{1, 20, 30, 40}, float32(2), float32(3), []float32{1, 2, 3, 3}, []float32{1, 2, 3}},
	{[]float64{1, 2, 3, 4}, []float64{1, 20, 30, 40}, float64(2), float64(3), []float64{1, 2, 3, 3}, []float64{1, 2, 3}},
}

func TestClamp(t *testing.T) {
	assert := assert.New(t)
	var got, sliced Tensor
	var T, reuse *Dense
	var err error
	for _, ct := range clampTests {
		T = New(WithBacking(ct.a))
		// safe
		if got, err = Clamp(T, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}
		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// sliced safe
		if sliced, err = T.Slice(makeRS(0, 3)); err != nil {
			t.Error("Unable to slice T")
			continue
		}
		if got, err = Clamp(sliced, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}

		// reuse
		reuse = New(WithBacking(ct.reuse))
		if got, err = Clamp(T, ct.min, ct.max, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}
		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// unsafe
		if got, err = Clamp(T, ct.min, ct.max, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(ct.correct, got.Data())
	}
}

func TestClampMasked(t *testing.T) {
	assert := assert.New(t)
	var got, sliced Tensor
	var T, reuse *Dense
	var err error
	for _, ct := range clampTestsMasked {
		T = New(WithBacking(ct.a, []bool{true, false, false, false}))
		// safe
		if got, err = Clamp(T, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}
		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// sliced safe
		if sliced, err = T.Slice(makeRS(0, 3)); err != nil {
			t.Error("Unable to slice T")
			continue
		}
		if got, err = Clamp(sliced, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}

		// reuse
		reuse = New(WithBacking(ct.reuse, []bool{true, false, false, false}))
		if got, err = Clamp(T, ct.min, ct.max, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}
		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// unsafe
		if got, err = Clamp(T, ct.min, ct.max, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(ct.correct, got.Data())
	}
}

var signTests = []struct {
	a, reuse      interface{}
	correct       interface{}
	correctSliced interface{}
}{
	{[]int{0, 1, 2, -2, -1}, []int{100, 10, 20, 30, 40}, []int{0, 1, 1, -1, -1}, []int{0, 1, 1, -1}},
	{[]int8{0, 1, 2, -2, -1}, []int8{100, 10, 20, 30, 40}, []int8{0, 1, 1, -1, -1}, []int8{0, 1, 1, -1}},
	{[]int16{0, 1, 2, -2, -1}, []int16{100, 10, 20, 30, 40}, []int16{0, 1, 1, -1, -1}, []int16{0, 1, 1, -1}},
	{[]int32{0, 1, 2, -2, -1}, []int32{100, 10, 20, 30, 40}, []int32{0, 1, 1, -1, -1}, []int32{0, 1, 1, -1}},
	{[]int64{0, 1, 2, -2, -1}, []int64{100, 10, 20, 30, 40}, []int64{0, 1, 1, -1, -1}, []int64{0, 1, 1, -1}},
	{[]float32{0, 1, 2, -2, -1}, []float32{100, 10, 20, 30, 40}, []float32{0, 1, 1, -1, -1}, []float32{0, 1, 1, -1}},
	{[]float64{0, 1, 2, -2, -1}, []float64{100, 10, 20, 30, 40}, []float64{0, 1, 1, -1, -1}, []float64{0, 1, 1, -1}},
}

var signTestsMasked = []struct {
	a, reuse interface{}
	correct  interface{}
	// correctSliced interface{}
}{
	{[]int{1, 2, -2, -1}, []int{10, 20, 30, 40}, []int{1, 1, -2, -1}},
	{[]int8{1, 2, -2, -1}, []int8{10, 20, 30, 40}, []int8{1, 1, -2, -1}},
	{[]int16{1, 2, -2, -1}, []int16{10, 20, 30, 40}, []int16{1, 1, -2, -1}},
	{[]int32{1, 2, -2, -1}, []int32{10, 20, 30, 40}, []int32{1, 1, -2, -1}},
	{[]int64{1, 2, -2, -1}, []int64{10, 20, 30, 40}, []int64{1, 1, -2, -1}},
	{[]float32{1, 2, -2, -1}, []float32{10, 20, 30, 40}, []float32{1, 1, -2, -1}},
	{[]float64{1, 2, -2, -1}, []float64{10, 20, 30, 40}, []float64{1, 1, -2, -1}},
}

func TestSign(t *testing.T) {
	assert := assert.New(t)
	var got, sliced Tensor
	var T, reuse *Dense
	var err error
	for _, st := range signTests {
		T = New(WithBacking(st.a))
		// safe
		if got, err = Sign(T); err != nil {
			t.Error(err)
			continue
		}

		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// sliced safe
		if sliced, err = T.Slice(makeRS(0, 4)); err != nil {
			t.Error("Unable to slice T")
			continue
		}
		if got, err = Sign(sliced); err != nil {
			t.Error(err)
			continue
		}
		assert.Equal(st.correctSliced, got.Data())

		// reuse
		reuse = New(WithBacking(st.reuse))
		if got, err = Sign(T, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}

		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// unsafe
		if got, err = Sign(T, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(st.correct, got.Data())
	}
}

func TestSignMasked(t *testing.T) {
	assert := assert.New(t)
	var got Tensor
	var T, reuse *Dense
	var err error
	for _, st := range signTestsMasked {
		T = New(WithBacking(st.a, []bool{false, false, true, false}))
		// safe
		if got, err = Sign(T); err != nil {
			t.Error(err)
			continue
		}

		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// reuse
		reuse = New(WithBacking(st.reuse, []bool{false, false, true, false}))
		if got, err = Sign(T, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}

		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// unsafe
		if got, err = Sign(T, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(st.correct, got.Data())
	}
}

var negTestsMasked = []struct {
	a, reuse interface{}
	correct  interface{}
}{
	{[]int{1, 2, -2, -1}, []int{10, 20, 30, 40}, []int{-1, -2, -2, 1}},
	{[]int8{1, 2, -2, -1}, []int8{10, 20, 30, 40}, []int8{-1, -2, -2, 1}},
	{[]int16{1, 2, -2, -1}, []int16{10, 20, 30, 40}, []int16{-1, -2, -2, 1}},
	{[]int32{1, 2, -2, -1}, []int32{10, 20, 30, 40}, []int32{-1, -2, -2, 1}},
	{[]int64{1, 2, -2, -1}, []int64{10, 20, 30, 40}, []int64{-1, -2, -2, 1}},
	{[]float32{1, 2, -2, -1}, []float32{10, 20, 30, 40}, []float32{-1, -2, -2, 1}},
	{[]float64{1, 2, -2, -1}, []float64{10, 20, 30, 40}, []float64{-1, -2, -2, 1}},
}

func TestNegMasked(t *testing.T) {
	assert := assert.New(t)
	var got Tensor
	var T, reuse *Dense
	var err error
	for _, st := range negTestsMasked {
		T = New(WithBacking(st.a, []bool{false, false, true, false}))
		// safe
		if got, err = Neg(T); err != nil {
			t.Error(err)
			continue
		}

		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// reuse
		reuse = New(WithBacking(st.reuse, []bool{false, false, true, false}))
		if got, err = Neg(T, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}

		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// unsafe
		if got, err = Neg(T, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(st.correct, got.Data())
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

func TestInv(t *testing.T) {
	var r *rand.Rand
	invFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Inv(a)
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		Mul(ret, a, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests for Inv failed: %v", err)
	}
}

func TestLog10(t *testing.T) {
	var r *rand.Rand
	invFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Log10(a)
		if err, retEarly := qcErrCheck(t, "Log10", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ten := identityVal(10, a.Dtype())
		Pow(ten, ret, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests for Log10 failed: %v", err)
	}
}

func TestAbs(t *testing.T) {
	var r *rand.Rand
	absFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		zeros := New(Of(q.Dtype()), WithShape(q.Shape().Clone()...))
		correct := New(Of(Bool), WithShape(q.Shape().Clone()...))
		correct.Memset(true)
		// we'll exclude everything other than ordtypes because complex numbers cannot be abs'd
		if err := typeclassCheck(a.Dtype(), ordTypes); err != nil {
			return true
		}
		we, willFailEq := willerr(a, signedTypes, nil)
		ret, err := Abs(a)
		if err, retEarly := qcErrCheck(t, "Abs", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		check, _ := Gte(ret, zeros)
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), check.Data()) {
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(absFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests for Abs failed: %v", err)
	}
}
