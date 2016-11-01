package tensorf32

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

func TestAt(t *testing.T) {
	backing := RangeFloat32(0, 6)
	T := NewTensor(WithShape(2, 3), WithBacking(backing))
	zeroone := T.At(0, 1)
	assert.Equal(t, float32(1), zeroone)

	oneone := T.At(1, 1)
	assert.Equal(t, float32(4), oneone)

	fail := func() {
		T.At(1, 2, 3)
	}
	assert.Panics(t, fail, "Expected too many coordinates to panic")

	backing = RangeFloat32(0, 24)
	T = NewTensor(WithShape(2, 3, 4), WithBacking(backing))
	/*
		T = [0, 1, 2, 3]
			[4, 5, 6, 7]
			[8, 9, 10, 11]

			[12, 13, 14, 15]
			[16, 17, 18, 19]
			[20, 21, 22, 23]
	*/
	oneoneone := T.At(1, 1, 1)
	assert.Equal(t, float32(17), oneoneone)
	zthreetwo := T.At(0, 2, 2)
	assert.Equal(t, float32(10), zthreetwo)
	onetwothree := T.At(1, 2, 3)
	assert.Equal(t, float32(23), onetwothree)

	fail = func() {
		T.At(0, 3, 2)
	}
	assert.Panics(t, fail)
}

func TestT_transposeIndex(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor

	T = NewTensor(WithShape(2, 2), WithBacking(RangeFloat32(0, 4)))

	correct := []int{0, 2, 1, 3}
	for i, v := range correct {
		assert.Equal(v, T.transposeIndex(i, []int{1, 0}, []int{2, 1}))
	}
}

var transposeTests = []struct {
	name          string
	shape         types.Shape
	transposeWith []int

	correctShape    types.Shape
	correctStrides  []int     // after .T()
	correctStrides2 []int     // after .Transpose()
	correctData     []float32 // after .Transpose()
}{
	{"c.T()", types.Shape{4, 1}, nil, types.Shape{1, 4}, []int{1}, []int{1}, RangeFloat32(0, 4)},
	{"r.T()", types.Shape{1, 4}, nil, types.Shape{4, 1}, []int{1}, []int{1}, RangeFloat32(0, 4)},
	{"v.T()", types.Shape{4}, nil, types.Shape{4}, []int{1}, []int{1}, RangeFloat32(0, 4)},
	{"M.T()", types.Shape{2, 3}, nil, types.Shape{3, 2}, []int{1, 3}, []int{2, 1}, []float32{0, 3, 1, 4, 2, 5}},
	{"M.T(0,1) (NOOP)", types.Shape{2, 3}, []int{0, 1}, types.Shape{2, 3}, []int{3, 1}, []int{3, 1}, RangeFloat32(0, 6)},
	{"3T.T()", types.Shape{2, 3, 4}, nil, types.Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1}, []float32{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},
	{"3T.T(2, 1, 0) (Same as .T())", types.Shape{2, 3, 4}, []int{2, 1, 0}, types.Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1}, []float32{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},
	{"3T.T(0, 2, 1)", types.Shape{2, 3, 4}, []int{0, 2, 1}, types.Shape{2, 4, 3}, []int{12, 1, 4}, []int{12, 3, 1}, []float32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}},
	{"3T.T{1, 0, 2)", types.Shape{2, 3, 4}, []int{1, 0, 2}, types.Shape{3, 2, 4}, []int{4, 12, 1}, []int{8, 4, 1}, []float32{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}},
	{"3T.T{1, 2, 0)", types.Shape{2, 3, 4}, []int{1, 2, 0}, types.Shape{3, 4, 2}, []int{4, 1, 12}, []int{8, 2, 1}, []float32{0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23}},
	{"3T.T{2, 0, 1)", types.Shape{2, 3, 4}, []int{2, 0, 1}, types.Shape{4, 2, 3}, []int{1, 12, 4}, []int{6, 3, 1}, []float32{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}},
	{"3T.T{0, 1, 2} (NOOP)", types.Shape{2, 3, 4}, []int{0, 1, 2}, types.Shape{2, 3, 4}, []int{12, 4, 1}, []int{12, 4, 1}, RangeFloat32(0, 24)},
}

func TestTranspose(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var err error

	// standard transposes
	for _, tts := range transposeTests {
		T = NewTensor(WithShape(tts.shape...), WithBacking(RangeFloat32(0, tts.shape.TotalSize())))
		if err = T.T(tts.transposeWith...); err != nil {
			t.Errorf("%v - %v", tts.name, err)
			continue
		}

		assert.True(tts.correctShape.Eq(T.Shape()), "Transpose %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T.Shape())
		assert.Equal(tts.correctStrides, T.Strides())
		T.Transpose()
		assert.True(tts.correctShape.Eq(T.Shape()), "Transpose %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T.Shape())
		assert.Equal(tts.correctStrides2, T.Strides())
		assert.Equal(tts.correctData, T.data)
	}

	// test stacked .T() calls

	// column vector
	T = NewTensor(WithShape(4, 1), WithBacking(RangeFloat32(0, 4)))
	if err = T.T(); err != nil {
		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
		goto matrev
	}
	if err = T.T(); err != nil {
		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
		goto matrev
	}
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)
	assert.True(T.IsColVec())

matrev:
	// matrix, reversed
	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat32(0, 6)))
	if err = T.T(); err != nil {
		t.Errorf("Stacked .T() #1 for matrix reverse. Error: %v", err)
		goto matnorev
	}
	if err = T.T(); err != nil {
		t.Errorf("Stacked .T() #2 for matrix reverse. Error: %v", err)
		goto matnorev
	}
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)
	assert.True(types.Shape{2, 3}.Eq(T.Shape()))

matnorev:
	// 3-tensor, non reversed
	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat32(0, 24)))
	if err = T.T(); err != nil {
		t.Fatalf("Stacked .T() #1 for tensor with no reverse. Error: %v", err)
	}
	if err = T.T(2, 0, 1); err != nil {
		t.Fatalf("Stacked .T() #2 for tensor with no reverse. Error: %v", err)
	}
	correctData := []float32{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}
	assert.Equal(correctData, T.data)
	assert.Equal([]int{2, 0, 1}, T.transposeWith)
	assert.NotNil(T.old)

}

func TestTUT(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor

	T = NewTensor(WithShape(2, 3, 4))
	T.T()
	T.UT()
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)

	T.T(2, 0, 1)
	T.UT()
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)
}

var repeatTestSlice = []struct {
	name                       string
	tensor                     *Tensor
	shouldAssertTensorNotEqual bool
	axis                       int
	repeats                    []int
	dataExpected               []float32
	shapeExpected              types.Shape
	isErrExpected              bool
	isPanicExpected            bool
}{
	{
		"Scalar repeats on axis 0",
		NewTensor(AsScalar(float32(3))),
		true,
		0,
		[]int{3},
		[]float32{3, 3, 3},
		types.Shape{3},
		false,
		false,
	},
	{
		"Scalar repeats on axis 1",
		NewTensor(AsScalar(float32(3))),
		false,
		1,
		[]int{3},
		[]float32{3, 3, 3},
		types.Shape{1, 3},
		false,
		false,
	},
	{
		"Vector repeats on axis 1: colvec",
		NewTensor(WithShape(2, 1), WithBacking([]float32{1, 2})),
		false,
		1,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{2, 3},
		false,
		false,
	},
	{
		"Vector repeats on axis 1:rowvec",
		NewTensor(WithShape(1, 2), WithBacking([]float32{1, 2})),
		false,
		1,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{1, 6},
		false,
		false,
	},
	{
		"Vector repeats on axis 0: vanilla vectors",
		NewTensor(WithShape(2), WithBacking([]float32{1, 2})),
		false,
		0,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{6},
		false,
		false,
	},
	{
		"Vector repeats on axis 0: colvec",
		NewTensor(WithShape(2, 1), WithBacking([]float32{1, 2})),
		false,
		0,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{6, 1},
		false,
		false,
	},
	{
		"Vector repeats on axis 0: rowvec",
		NewTensor(WithShape(1, 2), WithBacking([]float32{1, 2})),
		false,
		0,
		[]int{3},
		[]float32{1, 2, 1, 2, 1, 2},
		types.Shape{3, 2},
		false,
		false,
	},
	{
		"Vector repeats on axis -1: Shape(6) #1",
		NewTensor(WithShape(2, 1), WithBacking([]float32{1, 2})),
		false,
		-1,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{6},
		false,
		false,
	},
	{
		"Vector repeats on axis -1: Shape(6) #2",
		NewTensor(WithShape(1, 2), WithBacking([]float32{1, 2})),
		false,
		-1,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{6},
		false,
		false,
	},
	{
		"Vector repeats on axis -1: Shape(6) #3",
		NewTensor(WithShape(2), WithBacking([]float32{1, 2})),
		false,
		-1,
		[]int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		types.Shape{6},
		false,
		false,
	},
	{
		`Matrix:
		  1, 2,
		  3, 4`,
		NewTensor(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false,
		-1,
		[]int{1, 2, 1, 1},
		[]float32{1, 2, 2, 3, 4},
		types.Shape{5},
		false,
		false,
	},
	{
		`Matrix:
		  1, 1, 2,
		  3, 3, 4`,
		NewTensor(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false,
		1,
		[]int{2, 1},
		[]float32{1, 1, 2, 3, 3, 4},
		types.Shape{2, 3},
		false,
		false,
	},
	{
		`Matrix:
		  1, 2, 2,
		  3, 4, 4`,
		NewTensor(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false,
		1,
		[]int{1, 2},
		[]float32{1, 2, 2, 3, 4, 4},
		types.Shape{2, 3},
		false,
		false,
	},
	{
		`Matrix:
		  1, 2,
		  3, 4,
		  3, 4`,
		NewTensor(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false,
		0,
		[]int{1, 2},
		[]float32{1, 2, 3, 4, 3, 4},
		types.Shape{3, 2},
		false,
		false,
	},
	{
		`Matrix:
		  1, 2,
		  1, 2,
		  3, 4`,
		NewTensor(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false,
		0,
		[]int{2, 1},
		[]float32{1, 2, 1, 2, 3, 4},
		types.Shape{3, 2},
		false,
		false,
	},
	{
		`> 2D Matrix:
			In:
			1, 2,
			3, 4,
			5, 6,

			7, 8,
			9, 10,
			11, 12
		Out:
			1, 2,
			3, 4
			3, 4
			5, 6

			7, 8,
			9, 10,
			9, 10,
			11, 12`,
		NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat32(1, 2*3*2+1))),
		false,
		1,
		[]int{1, 2, 1},
		[]float32{1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12},
		types.Shape{2, 4, 2},
		false,
		false,
	},
	{
		"> 2D Matrix broadcast errors",
		NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat32(1, 2*3*2+1))),
		false,
		0,
		[]int{1, 2, 1},
		nil,
		nil,
		true,
		false,
	},
	{
		"> 2D Matrix generic repeat - repeat EVERYTHING by 2",
		NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat32(1, 2*3*2+1))),
		false,
		types.AllAxes,
		[]int{2},
		[]float32{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12},
		types.Shape{24},
		false,
		false,
	},
	{
		"> 2D Matrix generic repeat, axis specified",
		NewTensor(WithShape(2, 3, 2), WithBacking(RangeFloat32(1, 2*3*2+1))),
		false,
		2,
		[]int{2},
		[]float32{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12},
		types.Shape{2, 3, 4},
		false,
		false,
	},
	{
		"Repeat Scalars",
		NewTensor(AsScalar(float32(3))),
		false,
		0,
		[]int{5},
		[]float32{3, 3, 3, 3, 3},
		types.Shape{5},
		false,
		false,
	},
	{
		"IDIOTS SECTION - Trying to repeat on a nonexistent axis - Vector #1",
		NewTensor(WithShape(2, 1), WithBacking([]float32{1, 2})),
		false,
		2,
		[]int{3},
		nil,
		nil,
		false,
		true,
	},
	{
		"IDIOTS SECTION - Trying to repeat on a nonexistent axis - Vector #2",
		NewTensor(WithShape(2, 3), WithBacking([]float32{1, 2, 3, 4, 5, 6})),
		false,
		3,
		[]int{3},
		nil,
		nil,
		false,
		true,
	},
}

func TestTRepeat(t *testing.T) {
	assert := assert.New(t)

	for _, test := range repeatTestSlice {
		if test.isPanicExpected {
			fail := func() {
				test.tensor.Repeat(test.axis, test.repeats...)
			}
			assert.Panics(fail, test.name)
			continue
		}

		checkTensor, err := test.tensor.Repeat(test.axis, test.repeats...)
		if test.isErrExpected {
			assert.NotNil(err, test.name)
			continue
		}

		if test.shouldAssertTensorNotEqual {
			assert.NotEqual(test.tensor, checkTensor, test.name)
		}

		assert.Equal(test.dataExpected, checkTensor.data, test.name)
		assert.Equal(test.shapeExpected, checkTensor.Shape(), test.name)
	}
}

var sliceTests = []struct {
	name string

	shape  types.Shape
	slices []types.Slice

	correctShape  types.Shape
	correctStride []int
	correctData   []float32
}{
	{"a[0]", types.Shape{5}, []types.Slice{ss(0)}, types.ScalarShape(), nil, []float32{0}},
	{"a[0:2]", types.Shape{5}, []types.Slice{makeRS(0, 2)}, types.Shape{2}, []int{1}, []float32{0, 1}},
	{"a[1:5:2]", types.Shape{5}, []types.Slice{makeRS(1, 5, 2)}, types.Shape{2}, []int{2}, []float32{1, 2, 3, 4}},

	// colvec
	{"c[0]", types.Shape{5, 1}, []types.Slice{ss(0)}, types.ScalarShape(), nil, []float32{0}},
	{"c[0:2]", types.Shape{5, 1}, []types.Slice{makeRS(0, 2)}, types.Shape{2, 1}, []int{1}, []float32{0, 1}},
	{"c[1:5:2]", types.Shape{5, 1}, []types.Slice{makeRS(0, 5, 2)}, types.Shape{2, 1}, []int{2}, []float32{0, 1, 2, 3, 4}},

	// rowvec
	{"r[0]", types.Shape{1, 5}, []types.Slice{ss(0)}, types.Shape{1, 5}, []int{1}, []float32{0, 1, 2, 3, 4}},
	{"r[0:2]", types.Shape{1, 5}, []types.Slice{makeRS(0, 2)}, types.Shape{1, 5}, []int{1}, []float32{0, 1, 2, 3, 4}},
	{"r[0:5:2]", types.Shape{1, 5}, []types.Slice{makeRS(0, 5, 2)}, types.Shape{1, 5}, []int{1}, []float32{0, 1, 2, 3, 4}},
	{"r[:, 0]", types.Shape{1, 5}, []types.Slice{nil, ss(0)}, types.ScalarShape(), nil, []float32{0}},
	{"r[:, 0:2]", types.Shape{1, 5}, []types.Slice{nil, makeRS(0, 2)}, types.Shape{1, 2}, []int{1}, []float32{0, 1}},
	{"r[:, 1:5:2]", types.Shape{1, 5}, []types.Slice{nil, makeRS(1, 5, 2)}, types.Shape{1, 2}, []int{2}, []float32{1, 2, 3, 4}},

	// matrix
	{"A[0]", types.Shape{2, 3}, []types.Slice{ss(0)}, types.Shape{1, 3}, []int{1}, RangeFloat32(0, 3)},
	{"A[0:2]", types.Shape{4, 5}, []types.Slice{makeRS(0, 2)}, types.Shape{2, 5}, []int{5, 1}, RangeFloat32(0, 10)},
	{"A[0, 0]", types.Shape{4, 5}, []types.Slice{ss(0), ss(0)}, types.ScalarShape(), nil, []float32{0}},
	{"A[0, 1:5]", types.Shape{4, 5}, []types.Slice{ss(0), makeRS(1, 5)}, types.Shape{4}, []int{1}, RangeFloat32(1, 5)},
	{"A[0, 1:5:2]", types.Shape{4, 5}, []types.Slice{ss(0), makeRS(1, 5, 2)}, types.Shape{1, 2}, []int{2}, RangeFloat32(1, 5)},
	{"A[:, 0]", types.Shape{4, 5}, []types.Slice{nil, ss(0)}, types.Shape{4, 1}, []int{5}, RangeFloat32(0, 16)},
	{"A[:, 1:5]", types.Shape{4, 5}, []types.Slice{nil, makeRS(1, 5)}, types.Shape{4, 4}, []int{5, 1}, RangeFloat32(1, 20)},
	{"A[:, 1:5:2]", types.Shape{4, 5}, []types.Slice{nil, makeRS(1, 5, 2)}, types.Shape{4, 2}, []int{5, 2}, RangeFloat32(1, 20)},
}

func TestTSlice(t *testing.T) {
	assert := assert.New(t)
	var T, V *Tensor
	var err error

	for _, sts := range sliceTests {
		T = NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat32(0, sts.shape.TotalSize())))
		t.Log(sts.name)
		if V, err = T.Slice(sts.slices...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(V.Shape()), "Test: %v - Incorrect Shape. Correct: %v. Got %v", sts.name, sts.correctShape, V.Shape())
		assert.Equal(sts.correctStride, V.Strides(), "Test: %v - Incorrect Stride", sts.name)
		assert.Equal(sts.correctData, V.data, "Test: %v - Incorrect Data", sts.name)
	}

	// Transposed slice
	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat32(0, 6)))
	T.T()
	V, err = T.Slice(ss(0))
	assert.True(types.Shape{2}.Eq(V.Shape()))
	assert.Equal([]int{3}, V.Strides())
	assert.Equal([]float32{0, 1, 2, 3}, V.data)
	assert.Nil(V.old)

	// slice a sliced
	V, err = V.Slice(makeRS(1, 2))
	assert.True(types.ScalarShape().Eq(V.Shape()))
	assert.Equal([]float32{3}, V.data)

	// And now, ladies and gentlemen, the idiots!

	// too many slices
	_, err = T.Slice(ss(1), ss(2), ss(3), ss(4))
	if err == nil {
		t.Error("Expected a DimMismatchError error")
	}

	// out of range sliced
	_, err = T.Slice(makeRS(20, 5))
	if err == nil {
		t.Error("Expected a IndexError")
	}

	// surely nobody can be this dumb? Having a start of negatives
	_, err = T.Slice(makeRS(-1, 1))
	if err == nil {
		t.Error("Expected a IndexError")
	}

}

func TestT_at_itol(t *testing.T) {
	assert := assert.New(t)
	var err error
	var T *Tensor
	var shape types.Shape

	T = NewTensor(WithBacking(RangeFloat32(0, 12)), WithShape(3, 4))
	t.Logf("%+v", T)

	shape = T.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			coord := []int{i, j}
			idx, err := T.at(coord...)
			if err != nil {
				t.Error(err)
			}

			got, err := T.itol(idx)
			if err != nil {
				t.Error(err)
			}

			assert.Equal(coord, got)
		}
	}

	T = NewTensor(WithBacking(RangeFloat32(0, 24)), WithShape(2, 3, 4))

	shape = T.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			for k := 0; k < shape[2]; k++ {
				coord := []int{i, j, k}
				idx, err := T.at(coord...)
				if err != nil {
					t.Error(err)
				}

				got, err := T.itol(idx)
				if err != nil {
					t.Error(err)
				}

				assert.Equal(coord, got)
			}
		}
	}

	/* Transposes */

	T = NewTensor(WithBacking(RangeFloat32(0, 6)), WithShape(2, 3))
	t.Logf("%+v", T)
	err = T.T()
	if err != nil {
		t.Error(err)
	}
	t.Logf("%v, %v", T.Shape(), T.Shape())
	t.Logf("%v, %v", T.Strides(), T.ostrides())

	shape = T.Shape()
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			coord := []int{i, j}
			idx, err := T.at(coord...)
			if err != nil {
				t.Error(err)
				continue
			}

			got, err := T.itol(idx)
			if err != nil {
				t.Error(err)
				continue
			}

			assert.Equal(coord, got)
		}
	}

	/* IDIOT OF THE WEEK */

	T = NewTensor(WithBacking(RangeFloat32(0, 24)), WithShape(2, 3, 4))

	_, err = T.at(1, 3, 2) // the 3 is out of range
	if err == nil {
		t.Error("Expected an error")
	}
	t.Log(err)

	_, err = T.itol(24) // 24 is out of range
	if err == nil {
		t.Error("Expected an error")
	}
}

func TestCopyTo(t *testing.T) {
	assert := assert.New(t)
	var T, T2, T3 *Tensor
	var err error

	T = NewTensor(WithShape(2), WithBacking([]float32{1, 2}))
	T2 = NewTensor(WithShape(1, 2))

	err = T.CopyTo(T2)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(T2.data, T.data)

	// now, modify T1's data
	T.data[0] = 5000
	assert.NotEqual(T2.data, T.data)

	// test views
	T = NewTensor(WithShape(3, 3))
	T2 = NewTensor(WithShape(2, 2))
	T3, _ = T.Slice(makeRS(0, 2), makeRS(0, 2)) // T[0:2, 0:2], shape == (2,2)
	if err = T2.CopyTo(T3); err != nil {
		t.Log(err) // for now it's a not yet implemented error. TODO: FIX THIS
	}

	// dumbass time

	T = NewTensor(WithShape(3, 3))
	T2 = NewTensor(WithShape(2, 2))
	if err = T.CopyTo(T2); err == nil {
		t.Error("Expected an error")
	}

	if err = T.CopyTo(T); err != nil {
		t.Error("Copying a *Tensor to itself should yield no error. ")
	}

}

var concatTests = []struct {
	name  string
	shape types.Shape
	axis  int

	correctShape types.Shape
	correctData  []float32
}{
	{"vector", types.Shape{2}, 0, types.Shape{4}, []float32{0, 1, 0, 1}},
	{"matrix; axis 0 ", types.Shape{2, 2}, 0, types.Shape{4, 2}, []float32{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", types.Shape{2, 2}, 1, types.Shape{2, 4}, []float32{0, 1, 0, 1, 2, 3, 2, 3}},
}

func TestTensor_Concat(t *testing.T) {
	assert := assert.New(t)

	for _, cts := range concatTests {
		T0 := NewTensor(WithShape(cts.shape...), WithBacking(RangeFloat32(0, cts.shape.TotalSize())))
		T1 := NewTensor(WithShape(cts.shape...), WithBacking(RangeFloat32(0, cts.shape.TotalSize())))
		T2, err := T0.Concat(cts.axis, T1)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(cts.correctShape.Eq(T2.Shape()))
		assert.Equal(cts.correctData, T2.data)
	}
}

var simpleStackTests = []struct {
	name       string
	shape      types.Shape
	axis       int
	stackCount int

	correctShape types.Shape
	correctData  []float32
}{
	{"vector, axis 0, stack 2", types.Shape{2}, 0, 2, types.Shape{2, 2}, []float32{0, 1, 100, 101}},
	{"vector, axis 1, stack 2", types.Shape{2}, 1, 2, types.Shape{2, 2}, []float32{0, 100, 1, 101}},

	{"matrix, axis 0, stack 2", types.Shape{2, 3}, 0, 2, types.Shape{2, 2, 3}, []float32{0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105}},
	{"matrix, axis 1, stack 2", types.Shape{2, 3}, 1, 2, types.Shape{2, 2, 3}, []float32{0, 1, 2, 100, 101, 102, 3, 4, 5, 103, 104, 105}},
	{"matrix, axis 2, stack 2", types.Shape{2, 3}, 2, 2, types.Shape{2, 3, 2}, []float32{0, 100, 1, 101, 2, 102, 3, 103, 4, 104, 5, 105}},
	{"matrix, axis 0, stack 3", types.Shape{2, 3}, 0, 3, types.Shape{3, 2, 3}, []float32{0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205}},
	{"matrix, axis 1, stack 3", types.Shape{2, 3}, 1, 3, types.Shape{2, 3, 3}, []float32{0, 1, 2, 100, 101, 102, 200, 201, 202, 3, 4, 5, 103, 104, 105, 203, 204, 205}},
	{"matrix, axis 2, stack 3", types.Shape{2, 3}, 2, 3, types.Shape{2, 3, 3}, []float32{0, 100, 200, 1, 101, 201, 2, 102, 202, 3, 103, 203, 4, 104, 204, 5, 105, 205}},
}

var viewStackTests = []struct {
	name       string
	shape      types.Shape
	transform  []int
	slices     []types.Slice
	axis       int
	stackCount int

	correctShape types.Shape
	correctData  []float32
}{
	{"matrix(4x4)[1:3, 1:3] axis 0", types.Shape{4, 4}, nil, []types.Slice{makeRS(1, 3), makeRS(1, 3)}, 0, 2, types.Shape{2, 2, 2}, []float32{5, 6, 9, 10, 105, 106, 109, 110}},
	{"matrix(4x4)[1:3, 1:3] axis 1", types.Shape{4, 4}, nil, []types.Slice{makeRS(1, 3), makeRS(1, 3)}, 1, 2, types.Shape{2, 2, 2}, []float32{5, 6, 105, 106, 9, 10, 109, 110}},
	{"matrix(4x4)[1:3, 1:3] axis 2", types.Shape{4, 4}, nil, []types.Slice{makeRS(1, 3), makeRS(1, 3)}, 2, 2, types.Shape{2, 2, 2}, []float32{5, 105, 6, 106, 9, 109, 10, 110}},
}

func TestTensor_Stack(t *testing.T) {
	assert := assert.New(t)
	var err error
	for _, sts := range simpleStackTests {
		T := NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat32(0, sts.shape.TotalSize())))

		var stacked []*Tensor
		for i := 0; i < sts.stackCount-1; i++ {
			offset := (i + 1) * 100
			T1 := NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat32(offset, sts.shape.TotalSize()+offset)))
			stacked = append(stacked, T1)
		}

		T2, err := T.Stack(sts.axis, stacked...)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()))
		assert.Equal(sts.correctData, T2.data)
	}

	for _, sts := range viewStackTests {
		T := NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat32(0, sts.shape.TotalSize())))
		switch {
		case sts.slices != nil && sts.transform == nil:
			if T, err = T.Slice(sts.slices...); err != nil {
				t.Error(err)
				continue
			}
		case sts.transform != nil && sts.slices == nil:
			T.T(sts.transform...)
		}

		var stacked []*Tensor
		for i := 0; i < sts.stackCount-1; i++ {
			offset := (i + 1) * 100
			T1 := NewTensor(WithShape(sts.shape...), WithBacking(RangeFloat32(offset, sts.shape.TotalSize()+offset)))
			switch {
			case sts.slices != nil && sts.transform == nil:
				if T1, err = T1.Slice(sts.slices...); err != nil {
					t.Error(err)
					continue
				}
			case sts.transform != nil && sts.slices == nil:
				T1.T(sts.transform...)
			}

			stacked = append(stacked, T1)
		}
		T2, err := T.Stack(sts.axis, stacked...)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()))
		assert.Equal(sts.correctData, T2.data)
	}
}

var rollaxistests = []struct {
	axis, start int

	correctShape types.Shape
}{
	{0, 0, types.Shape{1, 2, 3, 4}},
	{0, 1, types.Shape{1, 2, 3, 4}},
	{0, 2, types.Shape{2, 1, 3, 4}},
	{0, 3, types.Shape{2, 3, 1, 4}},
	{0, 4, types.Shape{2, 3, 4, 1}},

	{1, 0, types.Shape{2, 1, 3, 4}},
	{1, 1, types.Shape{1, 2, 3, 4}},
	{1, 2, types.Shape{1, 2, 3, 4}},
	{1, 3, types.Shape{1, 3, 2, 4}},
	{1, 4, types.Shape{1, 3, 4, 2}},

	{2, 0, types.Shape{3, 1, 2, 4}},
	{2, 1, types.Shape{1, 3, 2, 4}},
	{2, 2, types.Shape{1, 2, 3, 4}},
	{2, 3, types.Shape{1, 2, 3, 4}},
	{2, 4, types.Shape{1, 2, 4, 3}},

	{3, 0, types.Shape{4, 1, 2, 3}},
	{3, 1, types.Shape{1, 4, 2, 3}},
	{3, 2, types.Shape{1, 2, 4, 3}},
	{3, 3, types.Shape{1, 2, 3, 4}},
	{3, 4, types.Shape{1, 2, 3, 4}},
}

// The RollAxis tests are directly adapted from Numpy's test cases.
func TestTensor_Rollaxis(t *testing.T) {
	assert := assert.New(t)
	var T *Tensor
	var err error

	for _, rats := range rollaxistests {
		T = NewTensor(WithShape(1, 2, 3, 4))
		if _, err = T.RollAxis(rats.axis, rats.start, false); assert.NoError(err) {
			assert.True(rats.correctShape.Eq(T.Shape()), "%d %d Expected %v, got %v", rats.axis, rats.start, rats.correctShape, T.Shape())
		}
	}
}
