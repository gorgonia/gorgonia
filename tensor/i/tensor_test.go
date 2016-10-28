package tensori

import (
	"testing"

	types "github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestCreate(t *testing.T) {
	assert := assert.New(t)

	t.Log("Standard, expected way of creating an ndarray")
	backingGood := make([]int, 2*2*6)
	T := NewTensor(WithShape(2, 2, 6), WithBacking(backingGood))

	expectedStrides := []int{12, 6, 1}
	assert.Equal(expectedStrides, T.Strides(), "Unequal strides")

	expectedDims := 3
	assert.Equal(expectedDims, T.Dims(), "Unequal dims")

	t.Log("Creating with just passing in a backing")
	T = NewTensor(WithBacking(backingGood)) // if you do this in real life without specifying a shap, you're an idiot
	expectedShape := types.Shape{len(backingGood)}
	assert.Equal(expectedShape, T.Shape(), "Unequal shape")

	t.Log("Creating with just a shape")
	T = NewTensor(WithShape(1, 3, 5))
	assert.Equal(15, T.Size(), "Unequal size")

	t.Log("Creating an ndarray with a mis match shape and elements")
	backingBad := []int{1, 2, 3, 4}
	badBackingF := func() {
		NewTensor(WithBacking(backingBad), WithShape(2, 2, 6))
	}
	assert.Panics(badBackingF, "Calling NewNDArray with bad backing should have panick'd")

	t.Logf("Making a scalar value a Tensor")
	T = NewTensor(AsScalar(3))
	assert.Equal(0, len(T.Shape()), "Expected a 1D shape")

	t.Log("Creating a ndarray with nothing passed in")
	noshapeF := func() {
		NewTensor()
	}
	assert.Panics(noshapeF, "Calling NewNDArray() without a shape should have panick'd")

}

func TestReshape(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var backing []int
	var err error

	t.Log("Testing standard reshape")
	backing = make([]int, 2*2*6)
	T = NewTensor(WithShape(2, 2, 6), WithBacking(backing))
	if err = T.Reshape(12, 2); err != nil {
		t.Errorf("There should be no error. Got %v instead", err)
	}

	expectedShape := types.Shape{12, 2}
	assert.Equal(expectedShape, T.Shape(), "Unequal shape")

	t.Log("Testing wrong reshape")
	if err = T.Reshape(12, 3); err == nil {
		t.Errorf("There should have been an error")
	}
}

func TestOnes(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var backing []int
	// var err error

	t.Log("Testing usual use case")
	backing = []int{1, 1, 1, 1}
	T = Ones(2, 2)

	expectedShape := types.Shape{2, 2}
	assert.Equal(expectedShape, T.Shape())
	assert.Equal(backing, T.data)

	t.Log("Testing stupid sizes: no size")
	T = Ones()
	assert.Nil(T.Shape())
	assert.Equal([]int{1}, T.data)
}

func TestClone(t *testing.T) {
	assert := assert.New(t)

	backing := []int{1, 2, 3, 4, 5, 6}
	T := NewTensor(WithBacking(backing), WithShape(2, 3))

	T1000 := T.Clone()
	// make sure that they are two different pointers, or else funny corruptions might happen
	if T.AP == T1000.AP {
		t.Error("Access Patterns must be two different objects")
	}
	// BUT the value must be the same
	assert.EqualValues(T.AP, T1000.AP)
	assert.Equal(T.data, T1000.data)

	// test transposes
	T.T()
	T1000 = T.Clone()
	if T.AP == T1000.AP {
		t.Error("AccessPatterns must be two different objects")
	}
	assert.EqualValues(T.AP, T1000.AP)
	assert.Equal(T.data, T1000.data)
	assert.EqualValues(T.old, T1000.old)
	assert.Equal(T.transposeWith, T1000.transposeWith)

	// TODO: test views
}

func TestI(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var correct []int

	T = I(4, 4, 0)
	correct = []int{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}
	assert.Equal(correct, T.data)

	T = I(4, 4, 1)
	correct = []int{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, 2)
	correct = []int{0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, 3)
	correct = []int{0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, 4)
	correct = []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, -1)
	correct = []int{0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, -2)
	correct = []int{0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, -3)
	correct = []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, -4)
	correct = []int{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	assert.Equal(correct, T.data)

	// non square identity (technically this shouldn't exist)
	T = I(4, 5, 0)
	correct = []int{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0}
	assert.Equal(correct, T.data)

	T = I(4, 5, 1)
	correct = []int{0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1}
	assert.Equal(correct, T.data)

	T = I(4, 5, -1)
	correct = []int{0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0}
	assert.Equal(correct, T.data)

	T = I(4, 4, -1)
	t.Logf("%+#v", T)
}

func TestAssignArray(t *testing.T) {
	assert := assert.New(t)
	T := NewTensor(WithShape(2, 2), WithBacking(RangeInt(0, 4)))
	T2 := NewTensor(WithShape(4, 4), WithBacking(RangeInt(0, 16)))
	S, _ := T2.Slice(makeRS(1, 3), makeRS(1, 3))

	err := assignArray(S, T)
	if err != nil {
		t.Error(err)
	}

	assert.Equal([]int{0, 1, 2, 3, 4, 0, 1, 7, 8, 2, 3, 11, 12, 13, 14, 15}, T2.data)

	// this should error? I dunno
	err = assignArray(T, T2)
	if err != nil {
		t.Error(err)
	}
	t.Logf("T: %+v", T)
	t.Logf("T2: %+v", T2)
}
