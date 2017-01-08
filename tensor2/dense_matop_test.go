package tensor

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

var atTests = []struct {
	data  Array
	shape Shape
	coord []int

	correct interface{}
	err     bool
}{
	// matrix
	{f64s{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{0, 1}, float64(1), false},
	{f32s{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{1, 1}, float32(4), false},
	{f64s{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{1, 2, 3}, nil, true},

	// 3-tensor
	{ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{1, 1, 1}, 17, false},
	{i64s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{1, 2, 3}, int64(23), false},
	{ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{0, 3, 2}, 23, true},
}

func TestDense_At(t *testing.T) {
	for i, ats := range atTests {
		T := New(WithShape(ats.shape...), WithBacking(ats.data))
		got, err := T.At(ats.coord...)

		switch {
		case ats.err:
			if err == nil {
				t.Error("Expected an error")
			}
			continue
		case !ats.err && err != nil:
			t.Errorf("i: %d Err: %v", i, err)
			continue
		}

		if got != ats.correct {
			t.Errorf("Expected %v. Got %v", ats.correct, got)
		}
	}
}

func Test_transposeIndex(t *testing.T) {
	a := u8s{0, 1, 2, 3}
	T := New(WithShape(2, 2), WithBacking(a))

	correct := []int{0, 2, 1, 3}
	for i, v := range correct {
		got := T.transposeIndex(i, []int{1, 0}, []int{2, 1})
		if v != got {
			t.Errorf("transposeIndex error. Expected %v. Got %v", v, got)
		}
	}
}

var transposeTests = []struct {
	name          string
	shape         Shape
	transposeWith []int
	data          Array

	correctShape    Shape
	correctStrides  []int // after .T()
	correctStrides2 []int // after .Transpose()
	correctData     Array
}{
	{"c.T()", Shape{4, 1}, nil, f64s{0, 1, 2, 3},
		Shape{1, 4}, []int{1}, []int{1}, f64s{0, 1, 2, 3}},

	{"r.T()", Shape{1, 4}, nil, f32s{0, 1, 2, 3},
		Shape{4, 1}, []int{1}, []int{1}, f32s{0, 1, 2, 3}},

	{"v.T()", Shape{4}, nil, ints{0, 1, 2, 3},
		Shape{4}, []int{1}, []int{1}, ints{0, 1, 2, 3}},

	{"M.T()", Shape{2, 3}, nil, i64s{0, 1, 2, 3, 4, 5},
		Shape{3, 2}, []int{1, 3}, []int{2, 1}, i64s{0, 3, 1, 4, 2, 5}},

	{"M.T(0,1) (NOOP)", Shape{2, 3}, []int{0, 1}, i32s{0, 1, 2, 3, 4, 5, 6},
		Shape{2, 3}, []int{3, 1}, []int{3, 1}, i32s{0, 1, 2, 3, 4, 5, 6}},

	{"3T.T()", Shape{2, 3, 4}, nil,
		u8s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},

		Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		u8s{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(2, 1, 0) (Same as .T())", Shape{2, 3, 4}, []int{2, 1, 0},
		ints{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		ints{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(0, 2, 1)", Shape{2, 3, 4}, []int{0, 2, 1},
		i32s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 4, 3}, []int{12, 1, 4}, []int{12, 3, 1},
		i32s{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}},

	{"3T.T{1, 0, 2)", Shape{2, 3, 4}, []int{1, 0, 2},
		f64s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{3, 2, 4}, []int{4, 12, 1}, []int{8, 4, 1},
		f64s{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}},

	{"3T.T{1, 2, 0)", Shape{2, 3, 4}, []int{1, 2, 0},
		f64s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{3, 4, 2}, []int{4, 1, 12}, []int{8, 2, 1},
		f64s{0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23}},

	{"3T.T{2, 0, 1)", Shape{2, 3, 4}, []int{2, 0, 1},
		f32s{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{4, 2, 3}, []int{1, 12, 4}, []int{6, 3, 1},
		f32s{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}},

	{"3T.T{0, 1, 2} (NOOP)", Shape{2, 3, 4}, []int{0, 1, 2},
		bs{true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false},
		Shape{2, 3, 4}, []int{12, 4, 1}, []int{12, 4, 1},
		bs{true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false}},
}

func TestDense_Transpose(t *testing.T) {
	assert := assert.New(t)
	var err error

	// standard transposes
	for _, tts := range transposeTests {
		T := New(WithShape(tts.shape...), WithBacking(tts.data))
		if err = T.T(tts.transposeWith...); err != nil {
			t.Errorf("%v - %v", tts.name, err)
			continue
		}

		assert.True(tts.correctShape.Eq(T.Shape()), "Transpose %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T.Shape())
		assert.Equal(tts.correctStrides, T.Strides())
		T.Transpose()
		assert.True(tts.correctShape.Eq(T.Shape()), "Transpose %v Expected shape: %v. Got %v", tts.name, tts.correctShape, T.Shape())
		assert.Equal(tts.correctStrides2, T.Strides())
		assert.Equal(tts.correctData, T.data, "Transpose %v", tts.name)
	}

	// test stacked .T() calls

	// 	// column vector
	// 	T = NewTensor(WithShape(4, 1), WithBacking(RangeFloat64(0, 4)))
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
	// 		goto matrev
	// 	}
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for vector. Error: %v", err)
	// 		goto matrev
	// 	}
	// 	assert.Nil(T.old)
	// 	assert.Nil(T.transposeWith)
	// 	assert.True(T.IsColVec())

	// matrev:
	// 	// matrix, reversed
	// 	T = NewTensor(WithShape(2, 3), WithBacking(RangeFloat64(0, 6)))
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #1 for matrix reverse. Error: %v", err)
	// 		goto matnorev
	// 	}
	// 	if err = T.T(); err != nil {
	// 		t.Errorf("Stacked .T() #2 for matrix reverse. Error: %v", err)
	// 		goto matnorev
	// 	}
	// 	assert.Nil(T.old)
	// 	assert.Nil(T.transposeWith)
	// 	assert.True(Shape{2, 3}.Eq(T.Shape()))

	// matnorev:
	// 	// 3-tensor, non reversed
	// 	T = NewTensor(WithShape(2, 3, 4), WithBacking(RangeFloat64(0, 24)))
	// 	if err = T.T(); err != nil {
	// 		t.Fatalf("Stacked .T() #1 for tensor with no reverse. Error: %v", err)
	// 	}
	// 	if err = T.T(2, 0, 1); err != nil {
	// 		t.Fatalf("Stacked .T() #2 for tensor with no reverse. Error: %v", err)
	// 	}
	// 	correctData := []float64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}
	// 	assert.Equal(correctData, T.data)
	// 	assert.Equal([]int{2, 0, 1}, T.transposeWith)
	// 	assert.NotNil(T.old)

}

func TestTUT(t *testing.T) {
	assert := assert.New(t)
	var T *Dense

	T = NewDense(Float64, Shape{2, 3, 4})
	T.T()
	T.UT()
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)

	T.T(2, 0, 1)
	T.UT()
	assert.Nil(T.old)
	assert.Nil(T.transposeWith)
}

var repeatTests = []struct {
	name                       string
	tensor                     *Tensor
	shouldAssertTensorNotEqual bool
	axis                       int
	repeats                    []int
	dataExpected               []float64
	shapeExpected              types.Shape
	isErrExpected              bool
}{}
