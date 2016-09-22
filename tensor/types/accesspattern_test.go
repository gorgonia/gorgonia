package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func dummyScalar1() *AP {
	return &AP{}
}

func dummyScalar2() *AP {
	return &AP{
		shape: Shape{1},
		dims:  0,
	}
}

func dummyColVec() *AP {
	return &AP{
		shape:   Shape{5, 1},
		strides: []int{1},
		dims:    1,
	}
}

func dummyRowVec() *AP {
	return &AP{
		shape:   Shape{1, 5},
		strides: []int{1},
		dims:    1,
	}
}

func dummyVec() *AP {
	return &AP{
		shape:   Shape{5},
		strides: []int{1},
		dims:    1,
	}
}

func twothree() *AP {
	return &AP{
		shape:   Shape{2, 3},
		strides: []int{3, 1},
		dims:    2,
	}
}

func twothreefour() *AP {
	return &AP{
		shape:   Shape{2, 3, 4},
		strides: []int{12, 4, 1},
		dims:    3,
	}
}

func TestAccessPatternIsX(t *testing.T) {
	assert := assert.New(t)
	var ap *AP

	ap = dummyScalar1()
	assert.True(ap.IsScalar())
	assert.False(ap.IsVector())
	assert.False(ap.IsColVec())
	assert.False(ap.IsRowVec())

	ap = dummyScalar2()
	assert.True(ap.IsScalar())
	assert.False(ap.IsVector())
	assert.False(ap.IsColVec())
	assert.False(ap.IsRowVec())

	ap = dummyColVec()
	assert.True(ap.IsColVec())
	assert.True(ap.IsVector())
	assert.False(ap.IsRowVec())
	assert.False(ap.IsScalar())

	ap = dummyRowVec()
	assert.True(ap.IsRowVec())
	assert.True(ap.IsVector())
	assert.False(ap.IsColVec())
	assert.False(ap.IsScalar())

	ap = twothree()
	assert.True(ap.IsMatrix())
	assert.False(ap.IsScalar())
	assert.False(ap.IsVector())
	assert.False(ap.IsRowVec())
	assert.False(ap.IsColVec())

}

func TestAccessPatternT(t *testing.T) {
	assert := assert.New(t)
	var ap, apT *AP
	var axes []int
	var err error

	ap = twothree()

	// test no axes
	apT, axes, err = ap.T()
	if err != nil {
		t.Error(err)
	}

	assert.Equal(Shape{3, 2}, apT.shape)
	assert.Equal([]int{1, 3}, apT.strides)
	assert.Equal([]int{1, 0}, axes)
	assert.Equal(2, apT.dims)

	// test no op
	apT, _, err = ap.T(0, 1)
	if err != nil {
		if _, ok := err.(NoOpError); !ok {
			t.Error(err)
		}
	}

	// test 3D
	ap = twothreefour()
	apT, axes, err = ap.T(2, 0, 1)
	if err != nil {
		t.Error(err)
	}
	assert.Equal(Shape{4, 2, 3}, apT.shape)
	assert.Equal([]int{1, 12, 4}, apT.strides)
	assert.Equal([]int{2, 0, 1}, axes)
	assert.Equal(3, apT.dims)

	// test stupid axes
	_, _, err = ap.T(1, 2, 3)
	if err == nil {
		t.Error("Expected an error")
	}
}

func TestTransposeIndex(t *testing.T) {
	var newInd int
	var oldShape Shape
	var pattern, oldStrides, newStrides, corrects []int

	/*
		(2,3)->(3,2)
		0, 1, 2
		3, 4, 5

		becomes

		0, 3
		1, 4
		2, 5

		1 -> 2
		2 -> 4
		3 -> 1
		4 -> 3
		0 and 5 stay the same
	*/

	oldShape = Shape{2, 3}
	pattern = []int{1, 0}
	oldStrides = []int{3, 1}
	newStrides = []int{2, 1}
	corrects = []int{0, 2, 4, 1, 3, 5}
	for i := 0; i < 6; i++ {
		newInd = TransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}

	/*
		(2,3,4) -(1,0,2)-> (3,2,4)
		0, 1, 2, 3
		4, 5, 6, 7
		8, 9, 10, 11

		12, 13, 14, 15
		16, 17, 18, 19
		20, 21, 22, 23

		becomes

		0,   1,  2,  3
		12, 13, 14, 15,

		4,   5,  6,  7
		16, 17, 18, 19

		8,   9, 10, 11
		20, 21, 22, 23
	*/
	oldShape = Shape{2, 3, 4}
	pattern = []int{1, 0, 2}
	oldStrides = []int{12, 4, 1}
	newStrides = []int{8, 4, 1}
	corrects = []int{0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23}
	for i := 0; i < len(corrects); i++ {
		newInd = TransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}

	/*
		(2,3,4) -(2,0,1)-> (4,2,3)
		0, 1, 2, 3
		4, 5, 6, 7
		8, 9, 10, 11

		12, 13, 14, 15
		16, 17, 18, 19
		20, 21, 22, 23

		becomes

		0,   4,  8
		12, 16, 20

		1,   5,  9
		13, 17, 21

		2,   6, 10
		14, 18, 22

		3,   7, 11
		15, 19, 23
	*/

	oldShape = Shape{2, 3, 4}
	pattern = []int{2, 0, 1}
	oldStrides = []int{12, 4, 1}
	newStrides = []int{6, 3, 1}
	corrects = []int{0, 6, 12, 18, 1, 7, 13, 19, 2, 8, 14, 20, 3, 9, 15, 21, 4, 10, 16, 22, 5, 11, 17, 23}
	for i := 0; i < len(corrects); i++ {
		newInd = TransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}

}

func TestUntransposeIndex(t *testing.T) {
	var newInd int
	var oldShape Shape
	var pattern, oldStrides, newStrides, corrects []int

	// vice versa
	oldShape = Shape{3, 2}
	oldStrides = []int{2, 1}
	newStrides = []int{3, 1}
	corrects = []int{0, 3, 1, 4, 2, 5}
	pattern = []int{1, 0}
	for i := 0; i < 6; i++ {
		newInd = UntransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}

	oldShape = Shape{3, 2, 4}
	oldStrides = []int{8, 4, 1}
	newStrides = []int{12, 4, 1}
	pattern = []int{1, 0, 2}
	corrects = []int{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}
	for i := 0; i < len(corrects); i++ {
		newInd = TransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}

	oldShape = Shape{4, 2, 3}
	pattern = []int{2, 0, 1}
	newStrides = []int{12, 4, 1}
	oldStrides = []int{6, 3, 1}
	corrects = []int{0, 4, 8, 12, 16, 20}
	for i := 0; i < len(corrects); i++ {
		newInd = UntransposeIndex(i, oldShape, pattern, oldStrides, newStrides)
		if newInd != corrects[i] {
			t.Errorf("Want %d, got %d instead", corrects[i], newInd)
		}
	}
}

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

	{Shape{2, 3, 4}, []int{12, 4, 1}, []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}}, // basic 3-Tensor
	{Shape{2, 4, 3}, []int{12, 1, 4}, []int{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}}, // basic 3-Tensor (under (0, 2, 1) transpose)
	{Shape{4, 2, 3}, []int{1, 12, 4}, []int{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}}, // basic 3-Tensor (under (2, 0, 1) transpose)
	{Shape{3, 2, 4}, []int{4, 12, 1}, []int{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}}, // basic 3-Tensor (under (1, 0, 2) transpose)
	{Shape{4, 3, 2}, []int{1, 4, 12}, []int{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}}, // basic 3-Tensor (under (2, 1, 0) transpose)
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
