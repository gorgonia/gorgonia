package tensorf64

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSafeVecAdd(t *testing.T) {
	a := RangeFloat64(0, 5)
	b := RangeFloat64(0, 10)
	fail := func() {
		safeVecAdd(a, b)
	}

	assert.Panics(t, fail, "Adding floats of different sizes should panic")

	b = RangeFloat64(0, 5)
	res := safeVecAdd(a, b)
	correct := []float64{0, 2, 4, 6, 8}
	assert.Equal(t, correct, res)
}
