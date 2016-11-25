package types

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShapeBasics(t *testing.T) {
	var s Shape
	var ds int
	var err error
	s = Shape{1, 2}

	if ds, err = s.DimSize(0); err != nil {
		t.Error(err)
	}
	if ds != 1 {
		t.Error("Expected DimSize(0) to be 1")
	}

	if ds, err = s.DimSize(2); err == nil {
		t.Error("Expected a DimensionMismatch error")
	}

	s = ScalarShape()
	if ds, err = s.DimSize(0); err != nil {
		t.Error(err)
	}

	if ds != 0 {
		t.Error("Expected DimSize(0) of a scalar to be 0")
	}

	// format for completeness sake
	s = Shape{2, 1}
	if fmt.Sprintf("%d", s) != "[2 1]" {
		t.Error("Shape.Format() error")
	}
}

func TestShapeIsX(t *testing.T) {
	assert := assert.New(t)
	var s Shape

	// scalar shape
	s = Shape{}
	assert.True(s.IsScalar())
	assert.False(s.IsVector())
	assert.False(s.IsColVec())
	assert.False(s.IsRowVec())

	s = Shape{1}
	assert.True(s.IsScalar())
	assert.False(s.IsVector())
	assert.False(s.IsColVec())
	assert.False(s.IsRowVec())

	// vector
	s = Shape{2}
	assert.False(s.IsScalar())
	assert.True(s.IsVector())
	assert.False(s.IsColVec())
	assert.False(s.IsRowVec())

	s = Shape{2, 1}
	assert.False(s.IsScalar())
	assert.True(s.IsVector())
	assert.True(s.IsColVec())
	assert.False(s.IsRowVec())

	s = Shape{1, 2}
	assert.False(s.IsScalar())
	assert.True(s.IsVector())
	assert.False(s.IsColVec())
	assert.True(s.IsRowVec())

	// matrix and up
	s = Shape{2, 2}
	assert.False(s.IsScalar())
	assert.False(s.IsVector())
	assert.False(s.IsColVec())
	assert.False(s.IsRowVec())
}

func TestShapeCalcStride(t *testing.T) {
	assert := assert.New(t)
	var s Shape

	// scalar shape
	s = Shape{}
	assert.Nil(s.CalcStrides())

	s = Shape{1}
	assert.Nil(s.CalcStrides())

	// vector shape
	s = Shape{2, 1}
	assert.Equal([]int{1}, s.CalcStrides())

	s = Shape{1, 2}
	assert.Equal([]int{1}, s.CalcStrides())

	s = Shape{2}
	assert.Equal([]int{1}, s.CalcStrides())

	// matrix strides
	s = Shape{2, 2}
	assert.Equal([]int{2, 1}, s.CalcStrides())

	s = Shape{5, 2}
	assert.Equal([]int{2, 1}, s.CalcStrides())

	// 3D strides
	s = Shape{2, 3, 4}
	assert.Equal([]int{12, 4, 1}, s.CalcStrides())

	// stupid shape
	s = Shape{-2, 1, 2}
	fail := func() {
		s.CalcStrides()
	}
	assert.Panics(fail)
}

func TestShapeEquality(t *testing.T) {
	assert := assert.New(t)
	var s1, s2 Shape

	// scalar
	s1 = Shape{1}
	s2 = Shape{}
	assert.True(s1.Eq(s2))
	assert.True(s2.Eq(s1))

	// vector
	s1 = Shape{3}
	s2 = Shape{5}
	assert.False(s1.Eq(s2))
	assert.False(s2.Eq(s1))

	s1 = Shape{2, 1}
	s2 = Shape{2, 1}
	assert.True(s1.Eq(s2))
	assert.True(s2.Eq(s1))

	s2 = Shape{2}
	assert.True(s1.Eq(s2))
	assert.True(s2.Eq(s1))

	s2 = Shape{1, 2}
	assert.False(s1.Eq(s2))
	assert.False(s2.Eq(s1))

	s1 = Shape{2}
	assert.True(s1.Eq(s2))
	assert.True(s2.Eq(s1))

	s2 = Shape{2, 3}
	assert.False(s1.Eq(s2))
	assert.False(s2.Eq(s1))

	// matrix
	s1 = Shape{2, 3}
	assert.True(s1.Eq(s2))
	assert.True(s2.Eq(s1))

	s2 = Shape{3, 2}
	assert.False(s1.Eq(s2))
	assert.False(s2.Eq(s1))

	// just for that green coloured code
	s1 = Shape{2}
	s2 = Shape{1, 3}
	assert.False(s1.Eq(s2))
	assert.False(s2.Eq(s1))
}

var shapeSliceTests = []struct {
	name string
	s    Shape
	sli  []Slice

	expected Shape
	err      bool
}{
	{"slicing a scalar shape", ScalarShape(), nil, ScalarShape(), false},
	{"slicing a scalar shape", ScalarShape(), []Slice{rs{0, 0, 0}}, nil, true},
	{"vec[0]", Shape{2}, []Slice{rs{0, 1, 0}}, ScalarShape(), false},
	{"vec[3]", Shape{2}, []Slice{rs{3, 4, 0}}, nil, true},
	{"vec[:, 0]", Shape{2}, []Slice{nil, rs{0, 1, 0}}, nil, true},
	{"vec[1:4:2]", Shape{5}, []Slice{rs{1, 4, 2}}, ScalarShape(), false},
}

func TestShape_Slice(t *testing.T) {
	for _, ssts := range shapeSliceTests {
		newShape, err := ssts.s.S(ssts.sli...)
		switch {
		case ssts.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !ssts.err && err != nil:
			t.Error(err)
			continue
		}

		if !ssts.expected.Eq(newShape) {
			t.Errorf("Test %q: Expeced shape %v. Got %v instead", ssts.name, ssts.expected, newShape)
		}
	}
}

var shapeRepeatTests = []struct {
	name    string
	s       Shape
	repeats []int
	axis    int

	expected        Shape
	expectedRepeats []int
	expectedSize    int
	err             bool
}{
	{"scalar repeat on axis 0", ScalarShape(), []int{3}, 0, Shape{3}, []int{3}, 1, false},
	{"scalar repeat on axis 1", ScalarShape(), []int{3}, 1, Shape{1, 3}, []int{3}, 1, false},
	{"vector repeat on axis 0", Shape{2}, []int{3}, 0, Shape{6}, []int{3, 3}, 2, false},
	{"vector repeat on axis 1", Shape{2}, []int{3}, 1, Shape{2, 3}, []int{3}, 1, false},
	{"colvec repeats on axis 0", Shape{2, 1}, []int{3}, 0, Shape{6, 1}, []int{3, 3}, 2, false},
	{"colvec repeats on axis 1", Shape{2, 1}, []int{3}, 1, Shape{2, 3}, []int{3}, 1, false},
	{"rowvec repeats on axis 0", Shape{1, 2}, []int{3}, 0, Shape{3, 2}, []int{3}, 1, false},
	{"rowvec repeats on axis 1", Shape{1, 2}, []int{3}, 1, Shape{1, 6}, []int{3, 3}, 2, false},
	{"3-Tensor repeats", Shape{2, 3, 2}, []int{1, 2, 1}, 1, Shape{2, 4, 2}, []int{1, 2, 1}, 3, false},
	{"3-Tensor generic repeats", Shape{2, 3, 2}, []int{2}, AllAxes, Shape{24}, []int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, 12, false},
	{"3-Tensor generic repeat, axis specified", Shape{2, 3, 2}, []int{2}, 2, Shape{2, 3, 4}, []int{2, 2}, 2, false},

	// stupids
	{"nonexisting axis 2", Shape{2, 1}, []int{3}, 2, nil, nil, 0, true},
	{"mismatching repeats", Shape{2, 3, 2}, []int{3, 1, 2}, 0, nil, nil, 0, true},
}

func TestShape_Repeat(t *testing.T) {
	assert := assert.New(t)
	for _, srts := range shapeRepeatTests {
		newShape, reps, size, err := srts.s.Repeat(srts.axis, srts.repeats...)

		switch {
		case srts.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !srts.err && err != nil:
			t.Error(err)
			continue
		}

		assert.True(srts.expected.Eq(newShape), "Test %q:  Want: %v. Got %v", srts.name, srts.expected, newShape)
		assert.Equal(srts.expectedRepeats, reps, "Test %q: ", srts.name)
		assert.Equal(srts.expectedSize, size, "Test %q: ", srts.name)
	}
}
