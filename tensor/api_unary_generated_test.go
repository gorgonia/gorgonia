package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

var clampTests = []struct {
	a, reuse interface{}
	min, max interface{}
	correct  interface{}
}{
	{[]int{1, 2, 3, 4}, []int{10, 20, 30, 40}, int(2), int(3), []int{2, 2, 3, 3}},
	{[]int8{1, 2, 3, 4}, []int8{10, 20, 30, 40}, int8(2), int8(3), []int8{2, 2, 3, 3}},
	{[]int16{1, 2, 3, 4}, []int16{10, 20, 30, 40}, int16(2), int16(3), []int16{2, 2, 3, 3}},
	{[]int32{1, 2, 3, 4}, []int32{10, 20, 30, 40}, int32(2), int32(3), []int32{2, 2, 3, 3}},
	{[]int64{1, 2, 3, 4}, []int64{10, 20, 30, 40}, int64(2), int64(3), []int64{2, 2, 3, 3}},
	{[]uint{1, 2, 3, 4}, []uint{10, 20, 30, 40}, uint(2), uint(3), []uint{2, 2, 3, 3}},
	{[]uint8{1, 2, 3, 4}, []uint8{10, 20, 30, 40}, uint8(2), uint8(3), []uint8{2, 2, 3, 3}},
	{[]uint16{1, 2, 3, 4}, []uint16{10, 20, 30, 40}, uint16(2), uint16(3), []uint16{2, 2, 3, 3}},
	{[]uint32{1, 2, 3, 4}, []uint32{10, 20, 30, 40}, uint32(2), uint32(3), []uint32{2, 2, 3, 3}},
	{[]uint64{1, 2, 3, 4}, []uint64{10, 20, 30, 40}, uint64(2), uint64(3), []uint64{2, 2, 3, 3}},
	{[]float32{1, 2, 3, 4}, []float32{10, 20, 30, 40}, float32(2), float32(3), []float32{2, 2, 3, 3}},
	{[]float64{1, 2, 3, 4}, []float64{10, 20, 30, 40}, float64(2), float64(3), []float64{2, 2, 3, 3}},
}

var clampTestsMasked = []struct {
	a, reuse interface{}
	min, max interface{}
	correct  interface{}
}{
	{[]int{1, 2, 3, 4}, []int{1, 20, 30, 40}, int(2), int(3), []int{1, 2, 3, 3}},
	{[]int8{1, 2, 3, 4}, []int8{1, 20, 30, 40}, int8(2), int8(3), []int8{1, 2, 3, 3}},
	{[]int16{1, 2, 3, 4}, []int16{1, 20, 30, 40}, int16(2), int16(3), []int16{1, 2, 3, 3}},
	{[]int32{1, 2, 3, 4}, []int32{1, 20, 30, 40}, int32(2), int32(3), []int32{1, 2, 3, 3}},
	{[]int64{1, 2, 3, 4}, []int64{1, 20, 30, 40}, int64(2), int64(3), []int64{1, 2, 3, 3}},
	{[]uint{1, 2, 3, 4}, []uint{1, 20, 30, 40}, uint(2), uint(3), []uint{1, 2, 3, 3}},
	{[]uint8{1, 2, 3, 4}, []uint8{1, 20, 30, 40}, uint8(2), uint8(3), []uint8{1, 2, 3, 3}},
	{[]uint16{1, 2, 3, 4}, []uint16{1, 20, 30, 40}, uint16(2), uint16(3), []uint16{1, 2, 3, 3}},
	{[]uint32{1, 2, 3, 4}, []uint32{1, 20, 30, 40}, uint32(2), uint32(3), []uint32{1, 2, 3, 3}},
	{[]uint64{1, 2, 3, 4}, []uint64{1, 20, 30, 40}, uint64(2), uint64(3), []uint64{1, 2, 3, 3}},
	{[]float32{1, 2, 3, 4}, []float32{1, 20, 30, 40}, float32(2), float32(3), []float32{1, 2, 3, 3}},
	{[]float64{1, 2, 3, 4}, []float64{1, 20, 30, 40}, float64(2), float64(3), []float64{1, 2, 3, 3}},
}

func TestClamp(t *testing.T) {
	assert := assert.New(t)
	var got Tensor
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
	var got Tensor
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
	a, reuse interface{}
	correct  interface{}
}{
	{[]int{1, 2, -2, -1}, []int{10, 20, 30, 40}, []int{1, 1, -1, -1}},
	{[]int8{1, 2, -2, -1}, []int8{10, 20, 30, 40}, []int8{1, 1, -1, -1}},
	{[]int16{1, 2, -2, -1}, []int16{10, 20, 30, 40}, []int16{1, 1, -1, -1}},
	{[]int32{1, 2, -2, -1}, []int32{10, 20, 30, 40}, []int32{1, 1, -1, -1}},
	{[]int64{1, 2, -2, -1}, []int64{10, 20, 30, 40}, []int64{1, 1, -1, -1}},
	{[]float32{1, 2, -2, -1}, []float32{10, 20, 30, 40}, []float32{1, 1, -1, -1}},
	{[]float64{1, 2, -2, -1}, []float64{10, 20, 30, 40}, []float64{1, 1, -1, -1}},
}

var signTestsMasked = []struct {
	a, reuse interface{}
	correct  interface{}
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
	var got Tensor
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
