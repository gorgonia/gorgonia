package tensor

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"runtime"
	"testing"
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

	// basic stuff
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

func TestFlatIteratorReverse(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var err error
	var nexts []int

	// basic stuff
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		err = nil
		ap = NewAP(fit.shape, fit.strides)
		it = NewFlatIterator(ap)
		it.SetReverse()
		for next, err := it.Next(); err == nil; next, err = it.Next() {
			nexts = append(nexts, next)
		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}
		// reverse slice
		for i, j := 0, len(nexts)-1; i < j; i, j = i+1, j-1 {
			nexts[i], nexts[j] = nexts[j], nexts[i]
		}
		// and then check
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}

func TestMultIterator(t *testing.T) {
	assert := assert.New(t)

	var ap []*AP
	var it *MultIterator
	var err error
	var nexts [][]int

	doReverse := []bool{false, true}
	for _, reverse := range doReverse {
		ap = make([]*AP, 6)
		nexts = make([][]int, 6)

		// Repeat flat tests
		for i, fit := range flatIterTests1 {
			nexts[0] = nexts[0][:0]
			err = nil
			ap[0] = NewAP(fit.shape, fit.strides)
			it = NewMultIterator(ap[0])
			if reverse {
				it.SetReverse()
			}
			for next, err := it.Next(); err == nil; next, err = it.Next() {
				nexts[0] = append(nexts[0], next)
			}
			if _, ok := err.(NoOpError); err != nil && !ok {
				t.Error(err)
			}
			if reverse {
				for i, j := 0, len(nexts[0])-1; i < j; i, j = i+1, j-1 {
					nexts[0][i], nexts[0][j] = nexts[0][j], nexts[0][i]
				}
			}
			assert.Equal(fit.correct, nexts[0], "Repeating flat test %d", i)
		}
		// Test multiple iterators simultaneously
		var choices = []int{0, 0, 9, 9, 0, 9}
		for j := 0; j < 6; j++ {
			fit := flatIterTests1[choices[j]]
			nexts[j] = nexts[j][:0]
			err = nil
			ap[j] = NewAP(fit.shape, fit.strides)
		}
		it = NewMultIterator(ap...)
		if reverse {
			it.SetReverse()
		}
		for _, err := it.Next(); err == nil; _, err = it.Next() {
			for j := 0; j < 6; j++ {
				nexts[j] = append(nexts[j], it.LastIndex(j))
			}

			if _, ok := err.(NoOpError); err != nil && !ok {
				t.Error(err)
			}
		}

		for j := 0; j < 6; j++ {
			fit := flatIterTests1[choices[j]]
			if reverse {
				for i, k := 0, len(nexts[j])-1; i < k; i, k = i+1, k-1 {
					nexts[j][i], nexts[j][k] = nexts[j][k], nexts[j][i]
				}
			}
			if ap[j].IsScalar() {
				assert.Equal(fit.correct, nexts[j][:1], "Test multiple iterators %d", j)
			} else {
				assert.Equal(fit.correct, nexts[j], "Test multiple iterators %d", j)
			}
		}
	}

}

func TestIteratorInterface(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it Iterator
	var err error
	var nexts []int

	// basic stuff
	for i, fit := range flatIterTests1 {
		nexts = nexts[:0]
		err = nil
		ap = NewAP(fit.shape, fit.strides)
		it = NewIterator(ap)
		for next, err := it.Start(); err == nil; next, err = it.Next() {
			nexts = append(nexts, next)
		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			t.Error(err)
		}
		assert.Equal(fit.correct, nexts, "Test %d", i)
	}
}

func TestMultIteratorFromDense(t *testing.T) {
	assert := assert.New(t)

	T1 := New(Of(Int), WithShape(3, 20))
	data1 := T1.Data().([]int)
	T2 := New(Of(Int), WithShape(3, 20))
	data2 := T2.Data().([]int)
	T3 := New(Of(Int), FromScalar(7))
	data3 := T3.Data().(int)

	for i := 0; i < 60; i++ {
		data1[i] = i
		data2[i] = 7 * i
	}
	it := MultIteratorFromDense(T1, T2, T3)
	runtime.SetFinalizer(it, destroyMultIterator)

	for _, err := it.Next(); err == nil; _, err = it.Next() {
		x := data1[it.LastIndex(0)]
		y := data2[it.LastIndex(1)]
		z := data3
		assert.True(y == x*z)
	}
}

func TestFlatIterator_Chan(t *testing.T) {
	assert := assert.New(t)

	var ap *AP
	var it *FlatIterator
	var nexts []int

	// basic stuff
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
	assert.Equal(0, it.nextIndex)
	assert.Equal(false, it.done)
	assert.Equal([]int{0, 0, 0}, it.track)

	for _, err := it.Next(); err == nil; _, err = it.Next() {
	}

	it.Reset()
	assert.Equal(0, it.nextIndex)
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

/*func BenchmarkOldFlatIterator(b *testing.B) {
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
}*/

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

func BenchmarkFlatIteratorParallel6(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := make([]*AP, 6)
	it := make([]*FlatIterator, 6)

	for j := 0; j < 6; j++ {
		ap[j] = NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})
		it[j] = NewFlatIterator(ap[j])
	}

	for n := 0; n < b.N; n++ {
		for _, err := it[0].Next(); err == nil; _, err = it[0].Next() {
			for j := 1; j < 6; j++ {
				it[j].Next()
			}

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}
		for j := 0; j < 6; j++ {
			it[j].Reset()
		}
	}

}

func BenchmarkFlatIteratorMulti1(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})

	it := NewMultIterator(ap)
	runtime.SetFinalizer(it, destroyMultIterator)

	for n := 0; n < b.N; n++ {
		for _, err := it.Next(); err == nil; _, err = it.Next() {

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}
		it.Reset()
	}
}

func BenchmarkFlatIteratorGeneric1(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})

	it := NewIterator(ap)
	runtime.SetFinalizer(it, destroyIterator)

	for n := 0; n < b.N; n++ {
		for _, err := it.Next(); err == nil; _, err = it.Next() {

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}
		it.Reset()
	}
}

func BenchmarkFlatIteratorMulti6(b *testing.B) {
	var err error

	// as if T = NewTensor(WithShape(30, 1000, 1000))
	// then T[:, 0:900:15, 250:750:50]
	ap := make([]*AP, 6)

	for j := 0; j < 6; j++ {
		ap[j] = NewAP(Shape{30, 60, 10}, []int{1000000, 15000, 50})
	}

	it := NewMultIterator(ap...)
	runtime.SetFinalizer(it, destroyMultIterator)

	for n := 0; n < b.N; n++ {
		for _, err := it.Next(); err == nil; _, err = it.Next() {

		}
		if _, ok := err.(NoOpError); err != nil && !ok {
			b.Error(err)
		}
		it.Reset()
	}
}
