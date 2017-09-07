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

func TestInvSqrt(t *testing.T) {
	var r *rand.Rand
	invFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := InvSqrt(a)
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		Sqrt(b, UseUnsafe())
		Mul(ret, b, UseUnsafe())
		if !qcEqCheck(t, b.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests for InvSqrt failed: %v", err)
	}

	// unsafe
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := InvSqrt(a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		Sqrt(b, UseUnsafe())
		Mul(ret, b, UseUnsafe())
		if !qcEqCheck(t, b.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for InvSqrt failed: %v", err)
	}

	// reuse
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		reuse := q.Clone().(*Dense)
		reuse.Zero()
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := InvSqrt(a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		Sqrt(b, UseUnsafe())
		Mul(ret, b, UseUnsafe())
		if !qcEqCheck(t, b.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be the same as reuse")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests with reuse for InvSqrt failed: %v", err)
	}

	// incr
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := InvSqrt(a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if ret, err = Sub(ret, identityVal(100, a.Dtype()), UseUnsafe()); err != nil {
			t.Errorf("err while subtracting incr: %v", err)
			return false
		}
		Sqrt(b, UseUnsafe())
		Mul(ret, b, UseUnsafe())
		if !qcEqCheck(t, b.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != incr {
			t.Errorf("Expected ret to be the same as incr")
			return false
		}
		return true
	}

	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests with incr for InvSqrt failed: %v", err)
	}

}

func TestInv(t *testing.T) {
	var r *rand.Rand
	invFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

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

	// unsafe
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Inv(a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		Mul(ret, b, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
	}

	// reuse
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		reuse := a.Clone().(*Dense)
		reuse.Zero()
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Inv(a, WithReuse(reuse))
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
		if ret != reuse {
			t.Errorf("Expected ret to be the same as reuse")
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
	}

	// incr
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Inv(a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if ret, err = Sub(ret, identityVal(100, a.Dtype()), UseUnsafe()); err != nil {
			t.Errorf("err while subtracting incr: %v", err)
			return false
		}
		Mul(ret, a, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != incr {
			t.Errorf("Expected ret to be the same as incr")
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
	}
}

func TestLog10(t *testing.T) {
	var r *rand.Rand

	// default
	invFn := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Log10er)
		we = we || !ok

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


	// unsafe
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Log10(a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ten := identityVal(10, a.Dtype())
		Pow(ten, b, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
	}


	// reuse
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		correct := a.Clone().(*Dense)
		reuse := a.Clone().(*Dense)
		reuse.Zero()
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Log10(a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
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
		if ret != reuse {
			t.Errorf("Expected ret to be the same as reuse")
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
	}

	// incr
	invFn = func(q *Dense) bool {
		a := q.Clone().(*Dense)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we, willFailEq := willerr(a, floatTypes, nil)
		_, ok := q.Engine().(Inver)
		we = we || !ok

		// we'll exclude everything other than floats
		if err := typeclassCheck(a.Dtype(), floatTypes); err != nil {
			return true
		}
		ret, err := Inv(a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Inv", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if ret, err = Sub(ret, identityVal(100, a.Dtype()), UseUnsafe()); err != nil {
			t.Errorf("err while subtracting incr: %v", err)
			return false
		}
		ten := identityVal(10, a.Dtype())
		Pow(ten, ret, UseUnsafe())
		if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
			return false
		}
		if ret != incr {
			t.Errorf("Expected ret to be the same as incr")
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(invFn, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv tests using unsafe for Inv failed: %v", err)
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
		_, ok := q.Engine().(Abser)
		we = we || !ok

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
