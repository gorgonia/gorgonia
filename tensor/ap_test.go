package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type dummySlice struct {
	start, end, step int
}

func (s dummySlice) Start() int { return s.start }
func (s dummySlice) End() int   { return s.end }
func (s dummySlice) Step() int  { return s.step }

func sli(start int, opt ...int) dummySlice {
	var end, step int
	switch len(opt) {
	case 0:
		end = start + 1
		step = 0
	case 1:
		end = opt[0]
		step = 1
	default:
		end = opt[0]
		step = opt[1]

	}
	return dummySlice{start: start, end: end, step: step}
}

func dummyScalar1() *AP {
	return &AP{}
}

func dummyScalar2() *AP {
	return &AP{
		shape: Shape{1},
	}
}

func dummyColVec() *AP {
	return &AP{
		shape:   Shape{5, 1},
		strides: []int{1},
	}
}

func dummyRowVec() *AP {
	return &AP{
		shape:   Shape{1, 5},
		strides: []int{1},
	}
}

func dummyVec() *AP {
	return &AP{
		shape:   Shape{5},
		strides: []int{1},
	}
}

func twothree() *AP {
	return &AP{
		shape:   Shape{2, 3},
		strides: []int{3, 1},
	}
}

func twothreefour() *AP {
	return &AP{
		shape:   Shape{2, 3, 4},
		strides: []int{12, 4, 1},
	}
}

func TestAccessPatternBasics(t *testing.T) {
	assert := assert.New(t)
	ap := new(AP)

	ap.SetShape(1, 2)
	assert.Equal(Shape{1, 2}, ap.Shape())
	assert.Equal([]int{1}, ap.Strides())
	assert.Equal(2, ap.Dims())
	assert.Equal(2, ap.Size())

	ap.SetShape(2, 3, 2)
	assert.Equal(Shape{2, 3, 2}, ap.Shape())
	assert.Equal([]int{6, 2, 1}, ap.Strides())
	assert.Equal(12, ap.Size())

	ap.lock()
	ap.SetShape(1, 2, 3)
	assert.Equal(Shape{2, 3, 2}, ap.shape)
	assert.Equal([]int{6, 2, 1}, ap.strides)

	ap.unlock()
	ap.SetShape(1, 2)
	assert.Equal(Shape{1, 2}, ap.Shape())
	assert.Equal([]int{1}, ap.Strides())
	assert.Equal(2, ap.Dims())
	assert.Equal(2, ap.Size())

	if ap.String() != "Shape: (1, 2), Stride: [1], Lock: false" {
		t.Error("AP formatting error. Got %q", ap.String())
	}

	ap2 := ap.Clone()
	assert.Equal(ap, ap2)
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
	assert.Equal(2, apT.Dims())

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
	assert.Equal(3, apT.Dims())

	// test stupid axes
	_, _, err = ap.T(1, 2, 3)
	if err == nil {
		t.Error("Expected an error")
	}
}

var sliceTests = []struct {
	name   string
	shape  Shape
	slices []Slice

	correctStart  int
	correctEnd    int
	correctShape  Shape
	correctStride []int
}{
	// vectors
	{"a[0]", Shape{5}, []Slice{sli(0)}, 0, 1, ScalarShape(), nil},
	{"a[0:2]", Shape{5}, []Slice{sli(0, 2)}, 0, 2, Shape{2}, []int{1}},
	{"a[1:3]", Shape{5}, []Slice{sli(1, 3)}, 1, 3, Shape{2}, []int{1}},
	{"a[1:5:2]", Shape{5}, []Slice{sli(1, 5, 2)}, 1, 5, Shape{2}, []int{2}},

	// matrix
	{"A[0]", Shape{2, 3}, []Slice{sli(0)}, 0, 3, Shape{1, 3}, []int{1}},
	{"A[1:3]", Shape{4, 5}, []Slice{sli(1, 3)}, 5, 15, Shape{2, 5}, []int{5, 1}},
	{"A[0:10] (intentionally over)", Shape{4, 5}, []Slice{sli(0, 10)}, 0, 20, Shape{4, 5}, []int{5, 1}}, // as if nothing happened

}

func TestAccessPatternS(t *testing.T) {
	assert := assert.New(t)
	var ap, apS *AP
	var ndStart, ndEnd int
	var err error

	for _, sts := range sliceTests {
		ap = NewAP(sts.shape, sts.shape.calcStrides())
		if apS, ndStart, ndEnd, err = ap.S(sts.shape.TotalSize(), sts.slices...); err != nil {
			t.Errorf("%v errored: %v", sts.name, err)
			continue
		}
		assert.Equal(sts.correctStart, ndStart, "Wrong start: %v. Want %d Got %d", sts.name, sts.correctStart, ndStart)
		assert.Equal(sts.correctEnd, ndEnd, "Wrong end: %v. Want %d Got %d", sts.name, sts.correctEnd, ndEnd)
		assert.True(sts.correctShape.Eq(apS.shape), "Wrong shape: %v. Want %v. Got %v", sts.name, sts.correctShape, apS.shape)
		assert.Equal(sts.correctStride, apS.strides, "Wrong strides: %v. Want %v. Got %v", sts.name, sts.correctStride, apS.strides)
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

func TestBroadcastStrides(t *testing.T) {
	ds := Shape{4, 4}
	ss := Shape{4}
	dst := []int{4, 1}
	sst := []int{1}

	st, err := BroadcastStrides(ds, ss, dst, sst)
	if err != nil {
		t.Error(err)
	}
	t.Log(st)
}
