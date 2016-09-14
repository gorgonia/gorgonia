package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

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
