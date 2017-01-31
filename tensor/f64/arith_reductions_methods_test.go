package tensorf64

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestTreduce(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var expectedShape types.Shape
	var expectedData []float64
	var err error

	/*
		3D tensor

		0, 1
		2, 3
		4, 5

		6, 7
		8, 9
		10, 11
	*/
	T = NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat64(0, 2*3*2)))
	T2, err = T.Reduce(add, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3, 2}
	expectedData = []float64{6, 8, 10, 12, 14, 16}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(add, 0, 1)
	if err != nil {
		t.Error(err)
	}

	expectedShape = types.Shape{2, 2}
	expectedData = []float64{6, 9, 24, 27}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(add, 0, 2)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{2, 3}
	expectedData = []float64{1, 5, 9, 13, 17, 21}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	/*
		Matrix

		0, 1, 2
		3, 4, 5
	*/
	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	T2, err = T.Reduce(add, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{3, 5, 7}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(mul, 1, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{0, 4, 10}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(div, 0, 0)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{3}
	expectedData = []float64{0, 0.25, 0.4}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	T2, err = T.Reduce(mul, 1, 1)
	if err != nil {
		t.Error(err)
	}
	expectedShape = types.Shape{2}
	expectedData = []float64{0, 60}
	assert.Equal(expectedShape, T2.Shape())
	assert.Equal(expectedData, T2.data)

	// idiotsville

	_, err = T.Reduce(add, 5, 10)
	assert.NotNil(err)
}

var sumTests = []struct {
	name  string
	shape types.Shape
	axis  int

	correctShape types.Shape
	correctData  []float64
}{
	// vector
	{"v.sum(0)", types.Shape{5}, 0, types.ScalarShape(), []float64{10}},
	{"c.sum(0)", types.Shape{5, 1}, 0, types.ScalarShape(), []float64{10}},
	{"c.sum(1)", types.Shape{5, 1}, 1, types.Shape{5}, []float64{0, 1, 2, 3, 4}},
	{"r.sum(0)", types.Shape{5, 1}, 0, types.ScalarShape(), []float64{10}},
	{"r.sum(1)", types.Shape{1, 5}, 1, types.ScalarShape(), []float64{10}},

	// matrix
	{"A.sum(0)", types.Shape{2, 2}, 0, types.Shape{2}, []float64{2, 4}},
	{"A.sum(1)", types.Shape{2, 2}, 1, types.Shape{2}, []float64{1, 5}},

	// 3Tensor
	{"3T.sum(0)", types.Shape{5, 3, 6}, 0, types.Shape{3, 6}, []float64{
		180, 185, 190, 195, 200, 205,
		210, 215, 220, 225, 230, 235,
		240, 245, 250, 255, 260, 265,
	}},
	{"3T.sum(1)", types.Shape{5, 3, 6}, 1, types.Shape{5, 6}, []float64{
		18, 21, 24, 27, 30, 33,
		72, 75, 78, 81, 84, 87,
		126, 129, 132, 135, 138, 141,
		180, 183, 186, 189, 192, 195,
		234, 237, 240, 243, 246, 249,
	}},
	{"3T.sum(2)", types.Shape{5, 3, 6}, 2, types.Shape{5, 3}, []float64{
		15, 51, 87,
		123, 159, 195,
		231, 267, 303,
		339, 375, 411,
		447, 483, 519,
	}},
}

func TestTsum(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	for _, sts := range sumTests {
		T = NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat64(0, sts.shape.TotalSize())))
		T2 = T.sum(sts.axis)
		assert.True(sts.correctShape.Eq(T2.Shape()), "Test %v - Correct shape is %v. Got %v", sts.name, sts.correctShape, T2.Shape())
		assert.Equal(sts.correctData, T2.data, "Test %v - wrong data", sts.name)
	}

	// scalar
	T = NewTensor(AsScalar(float64(5)))
	T2 = T.sum(0)
	assert.True(types.ScalarShape().Eq(T2.Shape()))
	assert.Equal(float64(5), T.data[0])
}

var SumTests = []struct {
	name  string
	along []int

	correctShape types.Shape
	correctData  []float64
}{
	// {"common case: T.Sum()", []int{}, types.ScalarShape(), []float64{15}},
	// {"A.Sum(0)", []int{0}, types.Shape{3}, []float64{3, 5, 7}},
	// {"A.Sum(1)", []int{1}, types.Shape{2}, []float64{3, 12}},
	// {"A.Sum(0,1)", []int{0, 1}, types.ScalarShape(), []float64{15}},
	{"A.Sum(1,0)", []int{1, 0}, types.ScalarShape(), []float64{15}},
}

func TestTSum(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var err error

	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	for _, sts := range SumTests {
		if T2, err = T.Sum(sts.along...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()), "Test %v. Correct shape is %v. Got %v", sts.name, sts.correctShape, T2.Shape())
		assert.Equal(sts.correctData, T2.data, "Test %v - wrong data", sts.name)
	}
	// T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	// if T2, err = T.Sum(1, 2); err != nil {
	// 	t.Error(err)
	// 	goto idiots
	// }
	// assert.True(types.Shape{2}.Eq(T2.Shape()), "3T.Max(1,2) error: Correct shape is (2). Got %v", T.Shape())
	// assert.Equal([]float64{66, 210}, T2.data)

	// idiots:
	// //	 IDIOT TESTING TIME
	// 	_, err = T.Sum(3)
	// 	assert.NotNil(err)
}

var maxTests = []struct {
	name   string
	shape  types.Shape
	modify []int
	axis   int

	correctShape types.Shape
	correctData  []float64
}{
	// vector
	{"v.max(0)", types.Shape{5}, []int{0, 3}, 0, types.ScalarShape(), []float64{2000}},
	{"c.max(0)", types.Shape{5, 1}, []int{0, 3}, 0, types.ScalarShape(), []float64{2000}},
	{"c.max(1)", types.Shape{5, 1}, []int{0, 3}, 1, types.Shape{5}, []float64{1000, 1, 2, 2000, 4}},
	{"r.max(0)", types.Shape{1, 5}, []int{0, 3}, 0, types.Shape{5}, []float64{1000, 1, 2, 2000, 4}},
	{"r.max(1)", types.Shape{1, 5}, []int{0, 3}, 1, types.ScalarShape(), []float64{2000}},

	// matrix
	{"A.max(0)", types.Shape{2, 3}, []int{0, 4}, 0, types.Shape{3}, []float64{1000, 2000, 5}},
	{"A.max(1)", types.Shape{2, 3}, []int{0, 4}, 1, types.Shape{2}, []float64{1000, 2000}},

	//3T
	{"3T.max(0)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 0, types.Shape{3, 4}, []float64{
		1000, 2000, 14, 15,
		16, 17, 3000, 4000,
		20, 21, 22, 23,
	}},

	{"3T.max(1)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 1, types.Shape{2, 4}, []float64{
		1000, 9, 3000, 11,
		20, 2000, 22, 4000,
	}},

	{"3T.max(2)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 2, types.Shape{2, 3}, []float64{
		1000, 3000, 11,
		2000, 4000, 23,
	}},
}

func TestTmax(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor

	for _, mts := range maxTests {
		T = NewTensor(WithShape(mts.shape...), WithBacking(RangeFloat64(0, mts.shape.TotalSize())))
		for i, ind := range mts.modify {
			T.data[ind] = float64((i + 1) * 1000)
		}
		T2 = T.max(mts.axis)
		assert.True(mts.correctShape.Eq(T2.Shape()), "Test %v - Correct shape is %v. Got %v", mts.name, mts.correctShape, T2.Shape())
		assert.Equal(mts.correctData, T2.data, "Test %v - wrong data", mts.name)
	}

	// scalar
	T = NewTensor(AsScalar(float64(5)))
	T2 = T.max(0)
	assert.True(types.ScalarShape().Eq(T2.Shape()))
	assert.Equal(float64(5), T.data[0])
}

var MaxTests = []struct {
	name  string
	along []int

	correctShape types.Shape
	correctData  []float64
}{
	{"common case: T.Max()", []int{}, types.ScalarShape(), []float64{1000}},
	{"A.Max(0)", []int{0}, types.Shape{3}, []float64{1000, 4, 5}},
	{"A.Max(1)", []int{1}, types.Shape{2}, []float64{1000, 5}},
	{"A.Max(0,1)", []int{0, 1}, types.ScalarShape(), []float64{1000}},
	{"A.Max(1,0)", []int{1, 0}, types.ScalarShape(), []float64{1000}},
}

func TestTMax(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var err error

	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	T.data[0] = 1000
	for _, mts := range MaxTests {
		if T2, err = T.Max(mts.along...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(mts.correctShape.Eq(T2.Shape()), "Test %v. Correct shape is %v. Got %v", mts.name, mts.correctShape, T2.Shape())
		assert.Equal(mts.correctData, T2.data, "Test %v - wrong data", mts.name)
	}

	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	if T2, err = T.Max(1, 2); err != nil {
		t.Error(err)
		goto idiots
	}
	assert.True(types.Shape{2}.Eq(T2.Shape()), "3T.Max(1,2) error: Correct shape is (2). Got %v", T.Shape())
	assert.Equal([]float64{11, 23}, T2.data)

idiots:
	/* IDIOT TESTING TIME */
	_, err = T.Max(5)
	assert.NotNil(err)
}

var minTests = []struct {
	name   string
	shape  types.Shape
	modify []int
	axis   int

	correctShape types.Shape
	correctData  []float64
}{
	// vector
	{"v.min(0)", types.Shape{5}, []int{0, 3}, 0, types.ScalarShape(), []float64{-2000}},
	{"c.min(0)", types.Shape{5, 1}, []int{0, 3}, 0, types.ScalarShape(), []float64{-2000}},
	{"c.min(1)", types.Shape{5, 1}, []int{0, 3}, 1, types.Shape{5}, []float64{-1000, 1, 2, -2000, 4}},
	{"r.min(0)", types.Shape{1, 5}, []int{0, 3}, 0, types.Shape{5}, []float64{-1000, 1, 2, -2000, 4}},
	{"r.min(1)", types.Shape{1, 5}, []int{0, 3}, 1, types.ScalarShape(), []float64{-2000}},

	// matrix
	{"A.min(0)", types.Shape{2, 3}, []int{0, 4}, 0, types.Shape{3}, []float64{-1000, -2000, 2}},
	{"A.min(1)", types.Shape{2, 3}, []int{0, 4}, 1, types.Shape{2}, []float64{-1000, -2000}},

	//3T
	{"3T.min(0)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 0, types.Shape{3, 4}, []float64{
		-1000, -2000, 2, 3,
		4, 5, -3000, -4000,
		8, 9, 10, 11,
	}},

	{"3T.min(1)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 1, types.Shape{2, 4}, []float64{
		-1000, 1, -3000, 3,
		12, -2000, 14, -4000,
	}},

	{"3T.min(2)", types.Shape{2, 3, 4}, []int{0, 13, 6, 19}, 2, types.Shape{2, 3}, []float64{
		-1000, -3000, 8,
		-2000, -4000, 20,
	}},
}

func TestTmin(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor

	for _, mts := range minTests {
		T = NewTensor(WithShape(mts.shape...), WithBacking(RangeFloat64(0, mts.shape.TotalSize())))
		for i, ind := range mts.modify {
			T.data[ind] = float64((i + 1) * -1000)
		}
		T2 = T.min(mts.axis)
		assert.True(mts.correctShape.Eq(T2.Shape()), "Test %v - Correct shape is %v. Got %v", mts.name, mts.correctShape, T2.Shape())
		assert.Equal(mts.correctData, T2.data, "Test %v - wrong data", mts.name)
	}

	// scalar
	T = NewTensor(AsScalar(float64(5)))
	T2 = T.min(0)
	assert.True(types.ScalarShape().Eq(T2.Shape()))
	assert.Equal(float64(5), T.data[0])
}

var MinTests = []struct {
	name  string
	along []int

	correctShape types.Shape
	correctData  []float64
}{
	{"common case: T.Min()", []int{}, types.ScalarShape(), []float64{-1000}},
	{"A.Min(0)", []int{0}, types.Shape{3}, []float64{-1000, 1, 2}},
	{"A.Min(1)", []int{1}, types.Shape{2}, []float64{-1000, 3}},
	{"A.Min(0,1)", []int{0, 1}, types.ScalarShape(), []float64{-1000}},
	{"A.Min(1,0)", []int{1, 0}, types.ScalarShape(), []float64{-1000}},
}

func TestTMin(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Tensor
	var err error

	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	T.data[0] = -1000
	for _, mts := range MinTests {
		if T2, err = T.Min(mts.along...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(mts.correctShape.Eq(T2.Shape()), "Test %v. Correct shape is %v. Got %v", mts.name, mts.correctShape, T2.Shape())
		assert.Equal(mts.correctData, T2.data, "Test %v - wrong data", mts.name)
	}

	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	if T2, err = T.Min(1, 2); err != nil {
		t.Error(err)
		goto idiots
	}
	assert.True(types.Shape{2}.Eq(T2.Shape()), "3T.Min(1,2) error: Correct shape is (2). Got %v", T.Shape())
	assert.Equal([]float64{0, 12}, T2.data)

idiots:
	/* IDIOT TESTING TIME */
	_, err = T.Min(5)
	assert.NotNil(err)
}
