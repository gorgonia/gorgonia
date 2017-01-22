package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var flatIterTests1 = []struct {
	shape   Shape
	strides []int

	correct []int
}{
	{ScalarShape(), []int{}, []int{0}},                  // scalar
	{Shape{5}, []int{1}, []int{0, 1, 2, 3, 4}},          // vector
	{Shape{5, 1}, []int{1}, []int{0, 1, 2, 3, 4}},       // colvec
	{Shape{1, 5}, []int{1}, []int{0, 1, 2, 3, 4}},       // rowvec
	{Shape{2, 3}, []int{3, 1}, []int{0, 1, 2, 3, 4, 5}}, // basic mat
	{Shape{3, 2}, []int{1, 3}, []int{0, 3, 1, 4, 2, 5}}, // basic mat, transposed
	{Shape{2}, []int{2}, []int{0, 2}},                   // basic 2x2 mat, sliced: Mat[:, 1]
	{Shape{2, 2}, []int{5, 1}, []int{0, 1, 5, 6}},       // basic 5x5, sliced: Mat[1:3, 2,4]
	{Shape{2, 2}, []int{1, 5}, []int{0, 5, 1, 6}},       // basic 5x5, sliced: Mat[1:3, 2,4] then transposed

	{Shape{2, 3, 4}, []int{12, 4, 1}, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}}, // basic 3-Tensor
	{Shape{2, 4, 3}, []int{12, 1, 4}, []int{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}}, // basic 3-Tensor (under (0, 2, 1) transpose)
	{Shape{4, 2, 3}, []int{1, 12, 4}, []int{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}}, // basic 3-Tensor (under (2, 0, 1) transpose)
	{Shape{3, 2, 4}, []int{4, 12, 1}, []int{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}}, // basic 3-Tensor (under (1, 0, 2) transpose)
	{Shape{4, 3, 2}, []int{1, 4, 12}, []int{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}}, // basic 3-Tensor (under (2, 1, 0) transpose)
}

var flatIterSlices = []struct {
	slices   []Slice
	corrects [][]int
}{
	{[]Slice{nil}, [][]int{{0}}},
	{[]Slice{rs{0, 3, 1}, rs{0, 5, 2}, rs{0, 6, -1}}, [][]int{{0, 1, 2}, {0, 2, 4}, {4, 3, 2, 1, 0}}},
}

func TestFlatIterator(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	var nexts []int

	// basic shit
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		err = nil
		ap = NewAP(fit.shape, fit.strides)
		it = NewFlatIterator(ap)
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			nexts = append(nexts, next)
		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}

func TestFlatIterator_Chan(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var nexts []int

	// basic shit
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		ap = NewAP(fit.shape, fit.strides)
		it = NewFlatIterator(ap)
		ch := it.Chan()
		for next := range ch {
			nexts = append(nexts, next)
		}
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}

func TestFlatIterator_Slice(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	var nexts []int

	for i, fit := range flatIterTests1 {
		ap = NewAP(fit.shape, fit.strides)
		it = NewFlatIterator(ap)
		nexts, err = it.Slice(nil)
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}

		assert.Equal(fit.correct, nexts, "Test %d", i)

		if i < len(flatIterSlices) {
			fis := flatIterSlices[i]
			for j, sli := range fis.slices {
				it.Reset()

				nexts, err = it.Slice(sli)
				if _, ok := err.(NoOpError); err != nil && !ok {
					t.Error(err)
				}

				assert.Equal(fis.corrects[j], nexts, "Test %d", i)
			}
		}
	}
}

func TestFlatIterator_Coord(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	// var nexts []int
	var donecount int

	ap = NewAP(Shape{2, 3, 4}, []int{12, 4, 1})
	it = NewFlatIterator(ap)

	var correct = [][]int{
		{0, 0, 1},
		{0, 0, 2},
		{0, 0, 3},
		{0, 1, 0},
		{0, 1, 1},
		{0, 1, 2},
		{0, 1, 3},
		{0, 2, 0},
		{0, 2, 1},
		{0, 2, 2},
		{0, 2, 3},
		{1, 0, 0},
		{1, 0, 1},
		{1, 0, 2},
		{1, 0, 3},
		{1, 1, 0},
		{1, 1, 1},
		{1, 1, 2},
		{1, 1, 3},
		{1, 2, 0},
		{1, 2, 1},
		{1, 2, 2},
		{1, 2, 3},
		{0, 0, 0},
	}

	for _, err = it.Next(); err == nil; _, err = it.Next() {
		assert.Equal(correct[donecount], it.Coord())
		donecount++
	}
}

// really this is just for completeness sake
func TestFlatIterator_Reset(t *testing.T) {
	assert := assert.New(t)
	ap := NewAP(Shape{2, 3, 4}, []int{12, 4, 1})
	it := NewFlatIterator(ap)

	it.Next()
	it.Next()
	it.Reset()
	assert.Equal(0, it.lastIndex)
	assert.Equal(false, it.done)
	assert.Equal([]int{0, 0, 0}, it.track)

	for _, err := it.Next(); err == nil; _, err = it.Next() {
	}

	it.Reset()
	assert.Equal(0, it.lastIndex)
	assert.Equal(false, it.done)
	assert.Equal([]int{0, 0, 0}, it.track)
}

/* BENCHMARK */
type oldFlatIterator struct {
	*AP

	//state
	lastIndex int
	track     []int
	done      bool
}

// NewFlatIterator creates a new FlatIterator
func newOldFlatIterator(ap *AP) *oldFlatIterator {
	return &oldFlatIterator{
		AP:    ap,
		track: make([]int, len(ap.shape)),
	}
}

func (it *oldFlatIterator) Next() (int, error) {
	if it.done {
		return -1, noopError{}
	}

	retVal, err := Ltoi(it.shape, it.strides, it.track...)
	it.lastIndex = retVal

	if it.IsScalar() {
		it.done = true
		return retVal, err
	}

	for d := len(it.shape) - 1; d >= 0; d-- {
		if d == 0 && it.track[0]+1 >= it.shape[0] {
			it.done = true
			it.track[d] = 0 // overflow it
			break
		}

		if it.track[d] < it.shape[d]-1 {
			it.track[d]++
			break
		}
		// overflow
		it.track[d] = 0
	}

	return retVal, err
}

func (it *oldFlatIterator) Reset() {
	it.done = false
	it.lastIndex = 0

	if it.done {
		return
	}

	for i := range it.track {
		it.track[i] = 0
	}
}

func BenchmarkOldFlatIterator(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})
	it := newOldFlatIterator(ap)

	for n := 0; n < b.N; n++ {
		for _, err := it.Next(); err == nil; _, err = it.Next() {

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}

		it.Reset()
	}
}

func BenchmarkFlatIterator(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})
	it := NewFlatIterator(ap)

	for n := 0; n < b.N; n++ {
		for _, err := it.Next(); err == nil; _, err = it.Next() {

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}

		it.Reset()
	}
}
