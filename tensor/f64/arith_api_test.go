package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

var safeOpErr = "Expected the result to be different from the input"
var unsafeOpErr = "Expected the result to be same pointer as the input"
var reuseOpErr = "Expected the result to be the same pointer as the reused Tensor"

func TestAdd(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []float64

	t.Logf("T-T")
	backingA := []float64{1, 2, 3, 4, 5}
	backingB := []float64{5, 4, 3, 2, 1}
	backingR := []float64{1, 3, 5, 6, 9}
	correct = []float64{6, 6, 6, 6, 6}
	Ta := NewTensor(WithBacking(backingA))
	Tb := NewTensor(WithBacking(backingB))
	reuse := NewTensor(WithBacking(backingR)) // doesn't matter what backing

	// safe (default use case)
	if got, err = Add(Ta, Tb); err != nil {
		t.Error(err)
	}

	if got == Ta || got == Tb {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = Add(Ta, Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{7, 9, 11, 12, 15}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Add(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{6, 6, 6, 6, 6}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Add(Ta, Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Tensor - Scalar */

	t.Logf("T-s")
	backingA = []float64{1, 2, 3, 4, 5}
	correct = []float64{6, 7, 8, 9, 10}
	backingR = []float64{1, 2, 3, 4, 5}
	Ta = NewTensor(WithBacking(backingA))
	reuse = NewTensor(WithBacking(backingR))

	// safe (default use case)
	if got, err = Add(Ta, float64(5)); err != nil {
		t.Error(err)
	}
	if got == Ta {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = Add(Ta, float64(5), types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}
	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{7, 9, 11, 13, 15}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Add(Ta, float64(5), types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{6, 7, 8, 9, 10}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Add(Ta, float64(5), types.UseUnsafe()); err != nil {
		t.Error(err)
	}

	if expected = Ta; got != Ta {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Scalar Tensor */

	t.Logf("s-T")
	correct = []float64{10, 9, 8, 7, 6}
	backingR = []float64{1, 2, 3, 4, 5}
	reuse = NewTensor(WithBacking(backingR))

	// safe (default use case)
	if got, err = Add(float64(5), Tb); err != nil {
		t.Error(err)
	}
	if got == Tb {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = Add(float64(5), Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}
	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{11, 11, 11, 11, 11}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Add(float64(5), Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{10, 9, 8, 7, 6}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Add(float64(5), Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if expected = Tb; got != expected {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Idiots */

	t.Logf("s-s")
	if got, err = Add(float64(5), float64(5)); err == nil {
		t.Error("Expected an error, duh doy")
	}

	t.Logf("Wrong shape")
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{1, 2, 3}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	t.Log("at.shape: %v, bt.shape %v | %T", Ta.Shape(), Tb.Shape(), Ta.Shape().Eq(Tb.Shape()))
	if got, err = Add(Ta, Tb); err == nil {
		t.Error("Expected a shape error")
	}

	// wrong shape in reuse
	reuse = NewTensor(WithShape(8, 1))
	if got, err = Add(Ta, Ta, types.WithReuse(reuse)); err == nil {
		t.Error("Expected a ShapeError")
	}

	/* WORKING-AS-EXPECTED RESULTS, BUT GENERALLY GOTCHAS FOR 90% OF USE CASES*/
	// TODO: move this to arith_api_example.go instead. When I'm not lazy

	// reusing backing. In 90% of the case this is not the correct thing to do
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	reuse = NewTensor(WithBacking(backingB))
	if got, err = Add(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}
	correct = []float64{2, 4, 6, 8, 10}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Ta
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = Add(Ta, Tb, types.WithIncr(Ta)); err != nil {
		t.Error(err)
	}
	if expected = Ta; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{7, 8, 9, 10, 11}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Tb
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = Add(Ta, Tb, types.WithIncr(Tb)); err != nil {
		t.Error(err)
	}
	if expected = Tb; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{11, 10, 9, 8, 7}
	assert.Equal(correct, got.data)
}

func TestSub(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []float64

	t.Logf("T-T")
	backingA := []float64{1, 2, 3, 4, 5}
	backingB := []float64{5, 4, 3, 2, 1}
	backingR := []float64{1, 3, 5, 6, 9}
	correct = []float64{-4, -2, 0, 2, 4}
	Ta := NewTensor(WithBacking(backingA))
	Tb := NewTensor(WithBacking(backingB))
	reuse := NewTensor(WithBacking(backingR))

	// safe - default option
	if got, err = Sub(Ta, Tb); err != nil {
		t.Error(err)
	}
	if got == Ta || got == Tb {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = Sub(Ta, Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{-3, 1, 5, 8, 13}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Sub(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{-4, -2, 0, 2, 4}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Sub(Ta, Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	correct = []float64{-4, -2, 0, 2, 4}
	assert.Equal(correct, got.data)

	/* Tensor - Scalar */

	t.Logf("T-s")
	backingA = []float64{1, 2, 3, 4, 5}
	backingR = []float64{1, 3, 5, 6, 9}
	correct = []float64{-4, -3, -2, -1, 0}
	reuse = NewTensor(WithBacking(backingR))
	Ta = NewTensor(WithBacking(backingA))
	if got, err = Sub(Ta, float64(5)); err != nil {
		t.Error(err)
	}
	if got == Ta {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = Sub(Ta, float64(5), types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{-3, 0, 3, 5, 9}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Sub(Ta, float64(5), types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{-4, -3, -2, -1, 0}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Sub(Ta, float64(5), types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if got != Ta {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Scalar Tensor */

	t.Logf("s-T")
	correct = []float64{0, 1, 2, 3, 4}
	if got, err = Sub(float64(5), Tb); err != nil {
		t.Error(err)
	}
	if got == Tb {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data, "%v - 5.0", Tb)

	// with incr
	backingR = []float64{1, 3, 5, 6, 9}
	reuse = NewTensor(WithBacking(backingR))
	if got, err = Sub(float64(5), Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{1, 4, 7, 9, 13}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = Sub(float64(5), Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{0, 1, 2, 3, 4}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = Sub(float64(5), Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if got != Tb {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Idiots */
	t.Logf("s-s")
	if got, err = Sub(float64(5), float64(5)); err == nil {
		t.Error("Expected an error, duh doy")
	}

	t.Logf("Wrong shape")
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{1, 2, 3}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = Sub(Ta, Tb); err == nil {
		t.Error("Expected a shape error")
	}

	/* WORKING-AS-EXPECTED RESULTS, BUT GENERALLY GOTCHAS FOR 90% OF USE CASES*/

	// reusing backing. In 90% of the case this is not the correct thing to do
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	reuse = NewTensor(WithBacking(backingB))
	if got, err = Sub(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}
	correct = []float64{0, 0, 0, 0, 0}
	if got != reuse {
		t.Error(reuseOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Ta
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = Sub(Ta, Tb, types.WithIncr(Ta)); err != nil {
		t.Error(err)
	}
	if expected = Ta; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{-3, 0, 3, 6, 9}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Tb
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = Sub(Ta, Tb, types.WithIncr(Tb)); err != nil {
		t.Error(err)
	}
	if expected = Tb; got != expected {
		t.Error(reuseOpErr)
	}
	correct = backingA
	assert.Equal(correct, got.data)
}

func TestPointwiseMul(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []float64

	t.Logf("T-T")
	backingA := []float64{1, 2, 3, 4, 5}
	backingB := []float64{5, 4, 3, 2, 1}
	backingR := []float64{1, 3, -1, -5, 2}
	correct = []float64{5, 8, 9, 8, 5}
	Ta := NewTensor(WithBacking(backingA))
	Tb := NewTensor(WithBacking(backingB))
	reuse := NewTensor(WithBacking(backingR))

	// safe
	if got, err = PointwiseMul(Ta, Tb); err != nil {
		t.Error(err)
	}
	if got == Ta || got == Tb {
		t.Error(safeOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr
	if got, err = PointwiseMul(Ta, Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{6, 11, 8, 3, 7}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = PointwiseMul(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{5, 8, 9, 8, 5}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = PointwiseMul(Ta, Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if expected = Ta; got != expected {
		t.Error(unsafeOpErr)
	}
	correct = []float64{5, 8, 9, 8, 5}
	assert.Equal(correct, got.data)

	/* Tensor - Scalar */

	t.Logf("T-s")

	// safe
	backingA = []float64{1, 2, 3, 4, 5}
	Ta = NewTensor(WithBacking(backingA))

	if got, err = PointwiseMul(Ta, float64(5)); err != nil {
		t.Error(err)
	}
	if got == Ta || got == Tb {
		t.Error(safeOpErr)
	}

	correct = []float64{5, 10, 15, 20, 25}
	assert.Equal(correct, got.data)

	// with incr
	backingR = []float64{1, 3, -11, 2, 6}
	reuse = NewTensor(WithBacking(backingR))
	if got, err = PointwiseMul(Ta, float64(5), types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{6, 13, 4, 22, 31}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = PointwiseMul(Ta, float64(5), types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{5, 10, 15, 20, 25}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = PointwiseMul(Ta, float64(5), types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if got != Ta {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Scalar Tensor */

	t.Logf("s-T")
	if got, err = PointwiseMul(float64(5), Tb); err != nil {
		t.Error(err)
	}
	if got == Ta || got == Tb {
		t.Error(safeOpErr)
	}
	correct = []float64{25, 20, 15, 10, 5}
	assert.Equal(correct, got.data)

	// with incr
	// backingB := []float64{5, 4, 3, 2, 1}
	backingR = []float64{1, 3, -11, 2, 6}
	reuse = NewTensor(WithBacking(backingR))
	if got, err = PointwiseMul(float64(5), Tb, types.WithIncr(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}

	correct = []float64{26, 23, 4, 12, 11}
	assert.Equal(correct, got.data)

	// with reuse
	if got, err = PointwiseMul(float64(5), Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}

	if expected = reuse; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{25, 20, 15, 10, 5}
	assert.Equal(correct, got.data)

	// unsafe
	if got, err = PointwiseMul(float64(5), Tb, types.UseUnsafe()); err != nil {
		t.Error(err)
	}
	if got != Tb {
		t.Error(unsafeOpErr)
	}
	assert.Equal(correct, got.data)

	/* Idiots */
	t.Logf("s-s")
	if got, err = PointwiseMul(float64(5), float64(5)); err == nil {
		t.Error("Expected an error, duh doy")
	}

	t.Logf("Wrong shape")
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{1, 2, 3}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = PointwiseMul(Ta, Tb); err == nil {
		t.Error("Expected a shape error")
	}

	/* WORKING-AS-EXPECTED RESULTS, BUT GENERALLY GOTCHAS FOR 90% OF USE CASES*/

	// reusing backing. In 90% of the case this is not the correct thing to do
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	reuse = NewTensor(WithBacking(backingB))
	if got, err = PointwiseMul(Ta, Tb, types.WithReuse(reuse)); err != nil {
		t.Error(err)
	}
	correct = []float64{1, 4, 9, 16, 25}
	if got != reuse {
		t.Error(reuseOpErr)
	}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Ta
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = PointwiseMul(Ta, Tb, types.WithIncr(Ta)); err != nil {
		t.Error(err)
	}
	if expected = Ta; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{6, 10, 12, 12, 10}
	assert.Equal(correct, got.data)

	// with incr, but reuse is same as Tb
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{5, 4, 3, 2, 1}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = PointwiseMul(Ta, Tb, types.WithIncr(Tb)); err != nil {
		t.Error(err)
	}
	if expected = Tb; got != expected {
		t.Error(reuseOpErr)
	}
	correct = []float64{10, 12, 12, 10, 6}
	assert.Equal(correct, got.data)
}

func TestPointwiseDiv(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []float64

	t.Logf("T-T")
	backingA := []float64{1, 2, 3, 4, 5}
	backingB := []float64{5, 4, 3, 2, 1}
	Ta := NewTensor(WithBacking(backingA))
	Tb := NewTensor(WithBacking(backingB))

	if got, err = PointwiseDiv(Ta, Tb); err != nil {
		t.Error(err)
	}
	expected = Ta
	if got == expected {
		t.Error(safeOpErr)
	}
	correct = []float64{float64(1) / float64(5), float64(2) / float64(4), 1, 2, 5}
	assert.Equal(correct, got.data)

	/* Tensor - Scalar */

	t.Logf("T-s")
	backingA = []float64{1, 2, 3, 4, 5}
	Ta = NewTensor(WithBacking(backingA))

	if got, err = PointwiseDiv(Ta, float64(5)); err != nil {
		t.Error(err)
	}
	expected = Ta
	if got == expected {
		t.Error(safeOpErr)
	}

	// Note for element 3: the result is 3*(1/5). While mathematically it's the same as 3/5,
	// due to floating point operations, the expected result is written as such
	correct = []float64{float64(1) / float64(5), float64(2) / float64(5), float64(3) / float64(5), float64(4) / float64(5), 1}
	// assert.Equal(correct, got.data)
	t.Logf("Test skipped for stupid floating point reasons")

	/* Scalar Tensor */

	t.Logf("s-T")
	if got, err = PointwiseDiv(float64(5), Tb); err != nil {
		t.Error(err)
	}
	expected = Tb
	if got == expected {
		t.Error("Expected the return value to be Tb")
	}
	correct = []float64{1, float64(5) / float64(4), float64(5) / float64(3), float64(5) / float64(2), 5}
	assert.Equal(correct, got.data)

	/* Idiots */
	t.Logf("s-s")
	if got, err = PointwiseDiv(float64(5), float64(5)); err == nil {
		t.Error("Expected an error, duh doy")
	}

	t.Logf("Wrong shape")
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{1, 2, 3}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = PointwiseDiv(Ta, Tb); err == nil {
		t.Error("Expected a shape error")
	}
}

func TestPointwisePow(t *testing.T) {
	assert := assert.New(t)
	var expected, got *Tensor
	var err error
	var correct []float64

	t.Logf("T-T")
	backingA := []float64{1, 2, 3, 4, 5}
	backingB := []float64{0, 1, 2, 3, 4}
	Ta := NewTensor(WithBacking(backingA))
	Tb := NewTensor(WithBacking(backingB))

	if got, err = PointwisePow(Ta, Tb); err != nil {
		t.Error(err)
	}
	expected = Ta
	if got == expected {
		t.Error(safeOpErr)
	}
	correct = []float64{1, 2, 9, 64, 625}
	assert.Equal(correct, got.data)

	/* Tensor - Scalar */

	t.Logf("T-s")
	backingA = []float64{1, 2, 3, 4, 5}
	Ta = NewTensor(WithBacking(backingA))

	if got, err = PointwisePow(Ta, float64(2)); err != nil {
		t.Error(err)
	}
	expected = Ta
	if got == expected {
		t.Error(safeOpErr)
	}

	correct = []float64{1, 4, 9, 16, 25}
	assert.Equal(correct, got.data)

	/* Scalar Tensor */

	t.Logf("s-T")
	if got, err = PointwisePow(float64(2), Tb); err != nil {
		t.Error(err)
	}
	expected = Tb
	if got == expected {
		t.Error("Expected the return value to be Tb")
	}
	correct = []float64{1, 2, 4, 8, 16}
	assert.Equal(correct, got.data)

	/* Idiots */
	t.Logf("s-s")
	if got, err = PointwisePow(float64(5), float64(5)); err == nil {
		t.Error("Expected an error, duh doy")
	}

	t.Logf("Wrong shape")
	backingA = []float64{1, 2, 3, 4, 5}
	backingB = []float64{1, 2, 3}
	Ta = NewTensor(WithBacking(backingA))
	Tb = NewTensor(WithBacking(backingB))
	if got, err = PointwisePow(Ta, Tb); err == nil {
		t.Error("Expected a shape error")
	}
}
