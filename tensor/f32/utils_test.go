package tensorf32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRangeFloat32(t *testing.T) {
	correct := []float32{0, 1, 2, 3, 4}
	actual := RangeFloat32(0, 5)
	assert.Equal(t, correct, actual)

	correct = []float32{1, 2, 3, 4, 5}
	actual = RangeFloat32(1, 6)
	assert.Equal(t, correct, actual)

	correct = []float32{5, 4, 3, 2, 1}
	actual = RangeFloat32(5, 0)
	assert.Equal(t, correct, actual)

	correct = []float32{-1, -2, -3, -4, -5}
	actual = RangeFloat32(-1, -6)
	assert.Equal(t, correct, actual)

	correct = []float32{3, 2, 1, 0, -1}
	actual = RangeFloat32(3, -2)
	assert.Equal(t, correct, actual)

}

func TestReduce(t *testing.T) {
	l := RangeFloat32(0, 4)
	res := reduce(add, 0, l...)
	if res != 6 {
		t.Error("Simple basic reduction fail")
	}

	// test with different default
	res = reduce(add, 1, l...)
	if res != 7 {
		t.Error("Simple basic reduction with different default fail")
	}

	res = reduce(add, 1)
	if res != 1 {
		t.Errorf("Reduction fail")
	}
}
