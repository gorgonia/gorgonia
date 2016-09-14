package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/gonum/matrix/mat64"
	"github.com/stretchr/testify/assert"
)

func TestFromMat64(t *testing.T) {
	assert := assert.New(t)
	var m *mat64.Dense
	var T *Tensor
	var backing []float64

	backing = RangeFloat64(0, 6)
	m = mat64.NewDense(2, 3, backing)

	T = FromMat64(m, true)
	assert.Equal(types.Shape{2, 3}, T.Shape())
	assert.Equal(backing, T.data)
	backing[0] = 1000
	assert.NotEqual(backing, T.data)

	backing[0] = 0
	T = FromMat64(m, false)
	backing[0] = 1000
	assert.Equal(backing, T.data)
}

func TestToMat64(t *testing.T) {
	assert := assert.New(t)
	var m *mat64.Dense
	var T *Tensor
	var backing []float64
	var err error

	backing = RangeFloat64(0, 6)

	T = NewTensor(WithShape(2, 3), WithBacking(backing))
	m, err = ToMat64(T, true)
	if err != nil {
		t.Fatal(err)
	}

	assert.Equal(backing, m.RawMatrix().Data)
	backing[0] = 1000
	assert.NotEqual(backing, m.RawMatrix().Data)

	backing[0] = 0
	m, err = ToMat64(T, false)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(backing, m.RawMatrix().Data)
	backing[0] = 1000
	assert.Equal(backing, m.RawMatrix().Data)

	// idiocy test
	T = NewTensor(WithShape(2, 3, 4))
	_, err = ToMat64(T, true)
	if err == nil {
		t.Error("Expected to have an error when trying to convert 3D matrix to mat64")
	}

}
