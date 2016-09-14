package tensori

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSafeVecAdd(t *testing.T) {
	a := RangeInt(0, 5)
	b := RangeInt(0, 10)
	fail := func() {
		safeVecAdd(a, b)
	}

	assert.Panics(t, fail, "Adding floats of different sizes should panic")

	b = RangeInt(0, 5)
	res := safeVecAdd(a, b)
	correct := []int{0, 2, 4, 6, 8}
	assert.Equal(t, correct, res)
}
