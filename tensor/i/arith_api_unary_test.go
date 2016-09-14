package tensori

import (
	"math"
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestPointwiseSquare(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []int

	backingA := []int{1, 2, -3, 4}
	backingR := []int{1, 3, 5, 6}
	correct = []int{1, 4, 9, 16}
	Ta := NewTensor(WithBacking(backingA))
	reuse := NewTensor(WithBacking(backingR)) // doesn't matter what backing

	// safe (default use case)
	if got, err = PointwiseSquare(Ta); err != nil {
		t.Error(err)
	}

	if got == Ta {
		t.Error(safeOpErr)
	}

	assert.Equal(correct, got.data)

	// with reuse
	if got, err = PointwiseSquare(Ta, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
		t.Errorf("%p %p %p", expected, reuse, got)
	}

	assert.Equal(correct, got.data)

	// unsafe
	if got, err = PointwiseSquare(Ta, types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Idiots */

	// wrong shape in reuse, same size
	reuse = NewTensor(WithShape(2, 2))
	if got, err = PointwiseSquare(Ta, types.WithReuse(reuse)); err != nil {
		t.Error("Different shapes, but same size, should work")
		t.Error(err)
	}

	// wrong size.
	reuse = NewTensor(WithShape(8, 1))
	if got, err = PointwiseSquare(Ta, types.WithReuse(reuse)); err == nil {
		t.Error("Expected ShapeError")
	}
}

func TestSqrt(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []int

	backingA := []int{1, 2, 3, 4}
	backingR := []int{1, 3, 5, 6}

	correct = make([]int, len(backingA))
	for i, v := range backingA {
		correct[i] = int(math.Sqrt(float64(v)))
	}

	Ta := NewTensor(WithBacking(backingA))
	reuse := NewTensor(WithBacking(backingR)) // doesn't matter what backing

	// safe (default use case)
	if got, err = Sqrt(Ta); err != nil {
		t.Error(err)
	}

	if got == Ta {
		t.Error(safeOpErr)
	}

	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Sqrt(Ta, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Sqrt(Ta, types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Idiots */

	// wrong shape in reuse, same size
	reuse = NewTensor(WithShape(2, 2))
	if got, err = Sqrt(Ta, types.WithReuse(reuse)); err != nil {
		t.Error("Different shapes, but same size, should work")
		t.Error(err)
	}

	// wrong size.
	reuse = NewTensor(WithShape(8, 1))
	if got, err = Sqrt(Ta, types.WithReuse(reuse)); err == nil {
		t.Error("Expected ShapeError")
	}
}

func TestClamp(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []int

	backingA := []int{1, 2, 3, 4}
	backingR := []int{1, 3, 5, 6}
	correct = []int{2, 2, 3, 3}

	Ta := NewTensor(WithBacking(backingA))
	reuse := NewTensor(WithBacking(backingR)) // doesn't matter what backing

	min := int(2)
	max := int(3)

	// safe
	if got, err = Clamp(Ta, min, max); err != nil {
		t.Error(err)
	}

	if got == Ta {
		t.Error(safeOpErr)
	}

	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Clamp(Ta, min, max, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Clamp(Ta, min, max, types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

}

func TestSign(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []int

	backingA := []int{1, 2, -2, -1}
	backingR := []int{1, 3, 5, 6}
	correct = []int{1, 1, -1, -1}

	Ta := NewTensor(WithBacking(backingA))
	reuse := NewTensor(WithBacking(backingR))

	// safe
	if got, err = Sign(Ta); err != nil {
		t.Error(err)
	}

	if got == Ta {
		t.Error(safeOpErr)
	}

	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Sign(Ta, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Sign(Ta, types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}

	assert.Equal(correct, got.data)
}
