package tensor

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestMaskedComparison(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4))
	assert.True(len(T.mask) < 1)
	assert.False(T.IsMasked())
	dataF64 := T.Data().([]float64)
	for i := range dataF64 {
		dataF64[i] = float64(i)
	}
	T.MaskedEqual(0.0)
	assert.True(T.IsMasked())
	T.MaskedEqual(1.0)
	assert.True(T.mask[0] && T.mask[1])
	T.MaskedNotEqual(2.0)
	assert.False(T.mask[2] && !(T.mask[0]))

	T.ResetMask()
	T.MaskedValues(3.0)
	assert.True(T.mask[3])

	T.ResetMask()
	T.MaskedInside(1.0, 22.0)
	assert.True(!T.mask[0] && !T.mask[23] && T.mask[1] && T.mask[22])

	T.ResetMask()
	T.MaskedOutside(1.0, 22.0)
	assert.True(T.mask[0] && T.mask[23] && !T.mask[1] && !T.mask[22])
}

func TestMaskedIteration(t *testing.T) {
	assert := assert.New(t)
	T := New(Of(Float64), WithShape(2, 3, 4, 5))
	assert.True(len(T.mask) < 1)
	dataF64 := T.Data().([]float64)
	for i := range dataF64 {
		dataF64[i] = float64(i)
	}
	for i := 0; i < 5; i++ {
		T.MaskedEqual(float64(i) * 10.0)
	}

	it := MultIteratorFromDense(T)

	j := 0
	for _, err := it.Next(); err == nil; _, err = it.Next() {
		j++
	}
	it.Reset()
	assert.True(j == 120)
	j = 0
	for _, err := it.NextValid(); err == nil; _, err = it.NextValid() {
		j++
	}
	it.Reset()
	assert.True(j == 115)
	j = 0
	for _, err := it.NextInvalid(); err == nil; _, err = it.NextInvalid() {
		j++
	}
	it.Reset()
	assert.True(j == 5)
}
