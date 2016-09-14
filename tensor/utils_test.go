package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// test case provided by a kind StackOverflow user: http://stackoverflow.com/questions/17901218/numpy-argsort-what-is-it-doing
func TestSortIndex(t *testing.T) {
	in := []float64{1.48, 1.41, 0.0, 0.1}
	out := SortIndex(in)

	expected := []int{2, 3, 1, 0}
	assert.Equal(t, expected, out, "Argsort is incorrect")
}
