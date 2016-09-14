package tensori

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestArgMax(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var backing []int
	var correct []int
	var maxes *Tensor
	var err error

	// Most basic (and indeed most common usecase)
	backing = RangeInt(0, 6)
	backing[1] = 10
	backing[3] = 30
	T = NewTensor(WithShape(2, 3), WithBacking(backing))
	if maxes, err = T.Argmax(-1); err != nil {
		t.Error(err)
	}
	if !maxes.IsScalar() {
		t.Fatal("Expected a scalar value")
	}
	assert.Equal(3, maxes.ScalarValue())

	/*
		0, 10, 2
		30, 4, 5

		argmax(0): [1,1,1]
		argmax(1): [1, 0]
	*/

	if maxes, err = T.Argmax(0); err != nil {
		t.Error(err)
	}
	correct = []int{1, 1, 1}
	assert.Equal(correct, maxes.Data())

	if maxes, err = T.Argmax(1); err != nil {
		t.Error(err)
	}
	correct = []int{1, 0}
	assert.Equal(correct, maxes.Data())

	/*
		0, 1, 2, 3
		4, 5, 6, 70
		8, 9, 100, 11

		12, 130, 14, 15
		160, 17, 18, 19
		20, 21, 22, 23

		argmax(0) =
			1, 1, 1
			1, 1, 0
			0, 1, 0
			1, 1, 1

		argmax(1) =
			2, 2, 1, 1
			0, 1, 2, 2

		argmax(2) =
			3, 3, 2
			0, 0, 3
	*/
	backing = RangeInt(0, 2*3*4)
	backing[7] = 70
	backing[10] = 100
	backing[12] = 130
	backing[16] = 160
	T = NewTensor(WithShape(2, 3, 4), WithBacking(backing))

	if maxes, err = T.Argmax(0); err != nil {
		t.Error(err)
	}
	correct = []int{1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1}
	assert.Equal(correct, maxes.Data())

	if maxes, err = T.Argmax(1); err != nil {
		t.Error(err)
	}
	correct = []int{2, 2, 1, 1, 0, 1, 2, 2}
	assert.Equal(correct, maxes.Data())

	if maxes, err = T.Argmax(2); err != nil {
		t.Error(err)
	}
	correct = []int{3, 3, 2, 0, 0, 3}
	assert.Equal(correct, maxes.Data())

}
