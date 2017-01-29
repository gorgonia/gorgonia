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
