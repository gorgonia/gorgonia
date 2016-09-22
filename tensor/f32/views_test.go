package tensorf32

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIterator(t *testing.T) {
	assert := assert.New(t)

	var T, v *Tensor
	var it *iterator
	var err error
	var nexts, correctNexts []int

	// slice a scalar
	T = NewTensor(WithShape(2), WithBacking([]float32{2, 1}))
	v, err = T.Slice(ss(0))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	assert.Equal([]int{0}, nexts)

	// slice a row vec
	T = NewTensor(WithBacking(RangeFloat32(0, 9)), WithShape(3, 3))
	v, err = T.Slice(makeRS(1, 2))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}

	assert.Equal([]int{0, 1, 2}, nexts)

	// slice a col vec
	v, err = T.Slice(nil, makeRS(1, 2))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	assert.Equal([]int{0, 3, 6}, nexts)

	// slice a submatrix
	v, err = T.Slice(makeRS(0, 2), makeRS(0, 2))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	assert.Equal([]int{0, 1, 3, 4}, nexts)

	// slice a submatrix
	v, err = T.Slice(ss(0), makeRS(1, 3))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	assert.Equal([]int{0, 1}, nexts)

	// 3D land
	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat32(0, 2*3*4)))

	// T[:, 1:3, :]
	v, err = T.Slice(nil, makeRS(1, 3))
	if err != nil {
		t.Error(err)
	}

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	correctNexts = []int{0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19}
	assert.Equal(correctNexts, nexts)

	// T[0, :, 2]
	v, err = T.Slice(ss(0), nil, ss(2))
	if err != nil {
		t.Error(err)
	}

	t.Logf("%+v", T)
	t.Logf("%+v", v.data)
	t.Logf("%+v", v.Shape())
	t.Logf("%v", v.ostrides())
	t.Logf("%v", T.ostrides())

	it = newIterator(v)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	correctNexts = []int{0, 4, 8}
	assert.Equal(correctNexts, nexts)

	/* Questionable things only questionable people will do */

	it = newIterator(T)
	nexts = nexts[:0]
	for next, err := it.next(); err == nil; next, err = it.next() {
		nexts = append(nexts, next)
	}
	correctNexts = correctNexts[:0]
	for i := range T.data {
		correctNexts = append(correctNexts, i)
	}
	assert.Equal(correctNexts, nexts)

}

func TestMaterialize(t *testing.T) {
	assert := assert.New(t)

	var T, T2, T3 *Tensor
	var err error

	T = NewTensor(WithShape(3, 3), WithBacking(RangeFloat32(0, 9)))
	T2, err = T.Slice(makeRS(0, 2), makeRS(0, 2)) // T[0:2, 0:2]
	if err != nil {
		t.Error(err)
	}

	T3 = T2.Materialize().(*Tensor)
	assert.Equal([]float32{0, 1, 3, 4}, T3.data)
	assert.Equal(T2.Shape(), T3.Shape())
	T2.data[0] = 5000
	assert.Equal([]float32{0, 1, 3, 4}, T3.data)

	// test materializing something that is not materializable
	T3 = T.Materialize().(*Tensor)
	assert.Equal(T, T3)
}
