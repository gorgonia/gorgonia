package tensor

import (
	//"fmt"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestDense_shallowClone(t *testing.T) {
	T := New(Of(Float64), WithBacking([]float64{1, 2, 3, 4}))
	T2 := T.shallowClone()
	T2.slice(0, 2)
	T2.float64s()[0] = 1000

	assert.Equal(t, T.Data().([]float64)[0:2], T2.Data())
}

func TestDenseMasked(t *testing.T) {
	T := New(Of(Float64), WithShape(3, 2), WithMaskStrides([]bool{true, true}))
	assert.Equal(t, T.mask, []bool{false, false, false, false, false, false})
	assert.Equal(t, T.maskStrides, []int{2, 1})

	T = New(Of(Float64), WithShape(3, 2), WithMaskStrides([]bool{true, false}))
	assert.Equal(t, T.mask, []bool{false, false, false})
	assert.Equal(t, T.maskStrides, []int{1, 0})

	T = New(Of(Float64), WithShape(3, 2), WithMaskStrides([]bool{false, true}))
	assert.Equal(t, T.mask, []bool{false, false})
	assert.Equal(t, T.maskStrides, []int{0, 1})

	T = New(Of(Float64), WithShape(3, 2), WithMaskStrides([]bool{false, false}))
	assert.Equal(t, len(T.mask), 0)
	assert.Equal(t, T.maskStrides, []int{0, 0})

}

func TestFromScalar(t *testing.T) {
	T := New(FromScalar(3.14))
	data := T.float64s()
	assert.Equal(t, []float64{3.14}, data)
}

func Test_recycledDense(t *testing.T) {
	T := recycledDense(Float64, ScalarShape())
	assert.Equal(t, float64(0), T.Data())
}
