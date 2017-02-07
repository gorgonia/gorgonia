package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDense_shallowClone(t *testing.T) {
	T := New(Of(Float64), WithBacking([]float64{1, 2, 3, 4}))
	T2 := T.shallowClone()
	T2.slice(0, 2)
	T2.float64s()[0] = 1000

	assert.Equal(t, T.Data().([]float64)[0:2], T2.Data())
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
