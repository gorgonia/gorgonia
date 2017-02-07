package tensor

import (
	"testing"

	"github.com/chewxy/vecf64"
	"github.com/stretchr/testify/assert"
)

func cloneArray(a interface{}) interface{} {
	switch at := a.(type) {
	case []float64:
		retVal := make([]float64, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []float32:
		retVal := make([]float32, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []int:
		retVal := make([]int, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []int64:
		retVal := make([]int64, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []int32:
		retVal := make([]int32, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []byte:
		retVal := make([]byte, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	case []bool:
		retVal := make([]bool, len(at))
		for i, v := range at {
			retVal[i] = v
		}
		return retVal
	}
	return nil
}

var atTests = []struct {
	data  interface{}
	shape Shape
	coord []int

	correct interface{}
	err     bool
}{
	// matrix
	{[]float64{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{0, 1}, float64(1), false},
	{[]float32{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{1, 1}, float32(4), false},
	{[]float64{0, 1, 2, 3, 4, 5}, Shape{2, 3}, []int{1, 2, 3}, nil, true},

	// 3-tensor
	{[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{1, 1, 1}, 17, false},
	{[]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{1, 2, 3}, int64(23), false},
	{[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 3, 4}, []int{0, 3, 2}, 23, true},
}

func TestDense_At(t *testing.T) {
	for i, ats := range atTests {
		T := New(WithShape(ats.shape...), WithBacking(ats.data))
		got, err := T.At(ats.coord...)
		if checkErr(t, ats.err, err, "At", i) {
			continue
		}

		if got != ats.correct {
			t.Errorf("Expected %v. Got %v", ats.correct, got)
		}
	}
}

func Test_transposeIndex(t *testing.T) {
	a := []byte{0, 1, 2, 3}
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
	data          interface{}

	correctShape    Shape
	correctStrides  []int // after .T()
	correctStrides2 []int // after .Transpose()
	correctData     interface{}
}{
	{"c.T()", Shape{4, 1}, nil, []float64{0, 1, 2, 3},
		Shape{1, 4}, []int{1}, []int{1}, []float64{0, 1, 2, 3}},

	{"r.T()", Shape{1, 4}, nil, []float32{0, 1, 2, 3},
		Shape{4, 1}, []int{1}, []int{1}, []float32{0, 1, 2, 3}},

	{"v.T()", Shape{4}, nil, []int{0, 1, 2, 3},
		Shape{4}, []int{1}, []int{1}, []int{0, 1, 2, 3}},

	{"M.T()", Shape{2, 3}, nil, []int64{0, 1, 2, 3, 4, 5},
		Shape{3, 2}, []int{1, 3}, []int{2, 1}, []int64{0, 3, 1, 4, 2, 5}},

	{"M.T(0,1) (NOOP)", Shape{2, 3}, []int{0, 1}, []int32{0, 1, 2, 3, 4, 5},
		Shape{2, 3}, []int{3, 1}, []int{3, 1}, []int32{0, 1, 2, 3, 4, 5}},

	{"3T.T()", Shape{2, 3, 4}, nil,
		[]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},

		Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		[]byte{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(2, 1, 0) (Same as .T())", Shape{2, 3, 4}, []int{2, 1, 0},
		[]int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{4, 3, 2}, []int{1, 4, 12}, []int{6, 2, 1},
		[]int{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},

	{"3T.T(0, 2, 1)", Shape{2, 3, 4}, []int{0, 2, 1},
		[]int32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{2, 4, 3}, []int{12, 1, 4}, []int{12, 3, 1},
		[]int32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}},

	{"3T.T{1, 0, 2)", Shape{2, 3, 4}, []int{1, 0, 2},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{3, 2, 4}, []int{4, 12, 1}, []int{8, 4, 1},
		[]float64{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}},

	{"3T.T{1, 2, 0)", Shape{2, 3, 4}, []int{1, 2, 0},
		[]float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{3, 4, 2}, []int{4, 1, 12}, []int{8, 2, 1},
		[]float64{0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23}},

	{"3T.T{2, 0, 1)", Shape{2, 3, 4}, []int{2, 0, 1},
		[]float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
		Shape{4, 2, 3}, []int{1, 12, 4}, []int{6, 3, 1},
		[]float32{0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23}},

	{"3T.T{0, 1, 2} (NOOP)", Shape{2, 3, 4}, []int{0, 1, 2},
		[]bool{true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false},
		Shape{2, 3, 4}, []int{12, 4, 1}, []int{12, 4, 1},
		[]bool{true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false}},

	{"M[2,2].T for bools, just for completeness sake", Shape{2, 2}, nil,
		[]bool{true, true, false, false},
		Shape{2, 2}, []int{1, 2}, []int{2, 1},
		[]bool{true, false, true, false},
	},
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
		assert.Equal(tts.correctData, T.Data(), "Transpose %v", tts.name)
	}

	// test stacked .T() calls
	var T *Dense

	// column vector
	T = New(WithShape(4, 1), WithBacking(Range(Int, 0, 4)))
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
	T = New(WithShape(2, 3), WithBacking(Range(Byte, 0, 6)))
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
	assert.True(Shape{2, 3}.Eq(T.Shape()))

matnorev:
	// 3-tensor, non reversed
	T = New(WithShape(2, 3, 4), WithBacking(Range(Int64, 0, 24)))
	if err = T.T(); err != nil {
		t.Fatalf("Stacked .T() #1 for tensor with no reverse. Error: %v", err)
	}
	if err = T.T(2, 0, 1); err != nil {
		t.Fatalf("Stacked .T() #2 for tensor with no reverse. Error: %v", err)
	}
	correctData := []int64{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}
	assert.Equal(correctData, T.Data())
	assert.Equal([]int{2, 0, 1}, T.transposeWith)
	assert.NotNil(T.old)

}

func TestTUT(t *testing.T) {
	assert := assert.New(t)
	var T *Dense

	T = New(Of(Float64), WithShape(2, 3, 4))
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
	name    string
	tensor  *Dense
	ne      bool // should assert tensor not equal
	axis    int
	repeats []int

	correct interface{}
	shape   Shape
	err     bool
}{
	{"Scalar Repeat on axis 0", New(FromScalar(true)),
		true, 0, []int{3},
		[]bool{true, true, true},
		Shape{3}, false,
	},

	{"Scalar Repeat on axis 1", New(FromScalar(byte(255))),
		false, 1, []int{3},
		[]byte{255, 255, 255},
		Shape{1, 3}, false,
	},

	{"Vector Repeat on axis 0", New(WithShape(2), WithBacking([]int32{1, 2})),
		false, 0, []int{3},
		[]int32{1, 1, 1, 2, 2, 2},
		Shape{6}, false,
	},

	{"ColVec Repeat on axis 0", New(WithShape(2, 1), WithBacking([]int64{1, 2})),
		false, 0, []int{3},
		[]int64{1, 1, 1, 2, 2, 2},
		Shape{6, 1}, false,
	},

	{"RowVec Repeat on axis 0", New(WithShape(1, 2), WithBacking([]int{1, 2})),
		false, 0, []int{3},
		[]int{1, 2, 1, 2, 1, 2},
		Shape{3, 2}, false,
	},

	{"ColVec Repeat on axis 1", New(WithShape(2, 1), WithBacking([]float32{1, 2})),
		false, 1, []int{3},
		[]float32{1, 1, 1, 2, 2, 2},
		Shape{2, 3}, false,
	},

	{"RowVec Repeat on axis 1", New(WithShape(1, 2), WithBacking([]float64{1, 2})),
		false, 1, []int{3},
		[]float64{1, 1, 1, 2, 2, 2},
		Shape{1, 6}, false,
	},

	{"Vector Repeat on all axes", New(WithShape(2), WithBacking([]byte{1, 2})),
		false, AllAxes, []int{3},
		[]byte{1, 1, 1, 2, 2, 2},
		Shape{6}, false,
	},

	{"ColVec Repeat on all axes", New(WithShape(2, 1), WithBacking([]int32{1, 2})),
		false, AllAxes, []int{3},
		[]int32{1, 1, 1, 2, 2, 2},
		Shape{6}, false,
	},

	{"RowVec Repeat on all axes", New(WithShape(1, 2), WithBacking([]int64{1, 2})),
		false, AllAxes, []int{3},
		[]int64{1, 1, 1, 2, 2, 2},
		Shape{6}, false,
	},

	{"M[2,2] Repeat on all axes with repeats = (1,2,1,1)", New(WithShape(2, 2), WithBacking([]int{1, 2, 3, 4})),
		false, AllAxes, []int{1, 2, 1, 1},
		[]int{1, 2, 2, 3, 4},
		Shape{5}, false,
	},

	{"M[2,2] Repeat on axis 1 with repeats = (2, 1)", New(WithShape(2, 2), WithBacking([]float32{1, 2, 3, 4})),
		false, 1, []int{2, 1},
		[]float32{1, 1, 2, 3, 3, 4},
		Shape{2, 3}, false,
	},

	{"M[2,2] Repeat on axis 1 with repeats = (1, 2)", New(WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4})),
		false, 1, []int{1, 2},
		[]float64{1, 2, 2, 3, 4, 4},
		Shape{2, 3}, false,
	},

	{"M[2,2] Repeat on axis 0 with repeats = (1, 2)", New(WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4})),
		false, 0, []int{1, 2},
		[]float64{1, 2, 3, 4, 3, 4},
		Shape{3, 2}, false,
	},

	{"M[2,2] Repeat on axis 0 with repeats = (2, 1)", New(WithShape(2, 2), WithBacking([]float64{1, 2, 3, 4})),
		false, 0, []int{2, 1},
		[]float64{1, 2, 1, 2, 3, 4},
		Shape{3, 2}, false,
	},

	{"3T[2,3,2] Repeat on axis 1 with repeats = (1,2,1)", New(WithShape(2, 3, 2), WithBacking(vecf64.Range(1, 2*3*2+1))),
		false, 1, []int{1, 2, 1},
		[]float64{1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 9, 10, 9, 10, 11, 12},
		Shape{2, 4, 2}, false,
	},

	{"3T[2,3,2] Generic Repeat by 2", New(WithShape(2, 3, 2), WithBacking(vecf64.Range(1, 2*3*2+1))),
		false, AllAxes, []int{2},
		[]float64{1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12},
		Shape{24}, false,
	},

	{"3T[2,3,2] repeat with broadcast errors", New(WithShape(2, 3, 2), WithBacking(vecf64.Range(1, 2*3*2+1))),
		false, 0, []int{1, 2, 1},
		nil, nil, true,
	},

	// idiots
	{"Nonexistent axis", New(WithShape(2, 1), WithBacking([]bool{true, false})),
		false, 2, []int{3}, nil, nil, true,
	},
}

func TestDense_Repeat(t *testing.T) {
	assert := assert.New(t)

	for i, test := range repeatTests {
		T, err := test.tensor.Repeat(test.axis, test.repeats...)
		if checkErr(t, test.err, err, "Repeat", i) {
			continue
		}

		var D *Dense
		if D, err = getDense(T); err != nil {
			t.Errorf("Expected Repeat to return a *Dense. got %v of %T instead", T, T)
			continue
		}

		if test.ne {
			assert.NotEqual(test.tensor, D, test.name)
		}

		assert.Equal(test.correct, D.Data(), test.name)
		assert.Equal(test.shape, D.Shape(), test.name)
	}
}

func TestDense_CopyTo(t *testing.T) {
	assert := assert.New(t)
	var T, T2 *Dense
	var T3 Tensor
	var err error

	T = New(WithShape(2), WithBacking([]float64{1, 2}))
	T2 = New(Of(Float64), WithShape(1, 2))

	err = T.CopyTo(T2)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(T2.Data(), T.Data())

	// now, modify T1's data
	T.Set(0, float64(5000))
	assert.NotEqual(T2.Data(), T.Data())

	// test views
	T = New(Of(Byte), WithShape(3, 3))
	T2 = New(Of(Byte), WithShape(2, 2))
	T3, _ = T.Slice(makeRS(0, 2), makeRS(0, 2)) // T[0:2, 0:2], shape == (2,2)
	if err = T2.CopyTo(T3.(*Dense)); err != nil {
		t.Log(err) // for now it's a not yet implemented error. TODO: FIX THIS
	}

	// dumbass time

	T = New(Of(Float32), WithShape(3, 3))
	T2 = New(Of(Float32), WithShape(2, 2))
	if err = T.CopyTo(T2); err == nil {
		t.Error("Expected an error")
	}

	if err = T.CopyTo(T); err != nil {
		t.Error("Copying a *Tensor to itself should yield no error. ")
	}

}

var denseSliceTests = []struct {
	name   string
	data   interface{}
	shape  Shape
	slices []Slice

	correctShape  Shape
	correctStride []int
	correctData   interface{}
}{
	{"a[0]", []bool{true, true, false, false, false},
		Shape{5}, []Slice{ss(0)}, ScalarShape(), nil, true},
	{"a[0:2]", Range(Byte, 0, 5), Shape{5}, []Slice{makeRS(0, 2)}, Shape{2}, []int{1}, []byte{0, 1}},
	{"a[1:5:2]", Range(Int32, 0, 5), Shape{5}, []Slice{makeRS(1, 5, 2)}, Shape{2}, []int{2}, []int32{1, 2, 3, 4}},

	// colvec
	{"c[0]", Range(Int64, 0, 5), Shape{5, 1}, []Slice{ss(0)}, ScalarShape(), nil, int64(0)},
	{"c[0:2]", Range(Float32, 0, 5), Shape{5, 1}, []Slice{makeRS(0, 2)}, Shape{2, 1}, []int{1}, []float32{0, 1}},
	{"c[1:5:2]", Range(Float64, 0, 5), Shape{5, 1}, []Slice{makeRS(0, 5, 2)}, Shape{2, 1}, []int{2}, []float64{0, 1, 2, 3, 4}},

	// rowvec
	{"r[0]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{ss(0)}, Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[0:2]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{makeRS(0, 2)}, Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[0:5:2]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{makeRS(0, 5, 2)}, Shape{1, 5}, []int{1}, []float64{0, 1, 2, 3, 4}},
	{"r[:, 0]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{nil, ss(0)}, ScalarShape(), nil, float64(0)},
	{"r[:, 0:2]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{nil, makeRS(0, 2)}, Shape{1, 2}, []int{1}, []float64{0, 1}},
	{"r[:, 1:5:2]", Range(Float64, 0, 5), Shape{1, 5}, []Slice{nil, makeRS(1, 5, 2)}, Shape{1, 2}, []int{2}, []float64{1, 2, 3, 4}},

	// matrix
	{"A[0]", Range(Float64, 0, 6), Shape{2, 3}, []Slice{ss(0)}, Shape{1, 3}, []int{1}, Range(Float64, 0, 3)},
	{"A[0:2]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{makeRS(0, 2)}, Shape{2, 5}, []int{5, 1}, Range(Float64, 0, 10)},
	{"A[0, 0]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{ss(0), ss(0)}, ScalarShape(), nil, float64(0)},
	{"A[0, 1:5]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{ss(0), makeRS(1, 5)}, Shape{4}, []int{1}, Range(Float64, 1, 5)},
	{"A[0, 1:5:2]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{ss(0), makeRS(1, 5, 2)}, Shape{1, 2}, []int{2}, Range(Float64, 1, 5)},
	{"A[:, 0]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{nil, ss(0)}, Shape{4, 1}, []int{5}, Range(Float64, 0, 16)},
	{"A[:, 1:5]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{nil, makeRS(1, 5)}, Shape{4, 4}, []int{5, 1}, Range(Float64, 1, 20)},
	{"A[:, 1:5:2]", Range(Float64, 0, 20), Shape{4, 5}, []Slice{nil, makeRS(1, 5, 2)}, Shape{4, 2}, []int{5, 2}, Range(Float64, 1, 20)},
}

func TestDense_Slice(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	var V Tensor
	var err error

	for _, sts := range denseSliceTests {
		T = New(WithShape(sts.shape...), WithBacking(sts.data))
		t.Log(sts.name)
		if V, err = T.Slice(sts.slices...); err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(V.Shape()), "Test: %v - Incorrect Shape. Correct: %v. Got %v", sts.name, sts.correctShape, V.Shape())
		assert.Equal(sts.correctStride, V.Strides(), "Test: %v - Incorrect Stride", sts.name)
		assert.Equal(sts.correctData, V.Data(), "Test: %v - Incorrect Data", sts.name)
	}

	// Transposed slice
	T = New(WithShape(2, 3), WithBacking(Range(Float32, 0, 6)))
	T.T()
	V, err = T.Slice(ss(0))
	assert.True(Shape{2}.Eq(V.Shape()))
	assert.Equal([]int{3}, V.Strides())
	assert.Equal([]float32{0, 1, 2, 3}, V.Data())
	assert.Nil(V.(*Dense).old)

	// slice a sliced
	V, err = V.Slice(makeRS(1, 2))
	assert.True(ScalarShape().Eq(V.Shape()))
	assert.Equal(float32(3), V.Data())

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

var rollaxisTests = []struct {
	axis, start int

	correctShape Shape
}{
	{0, 0, Shape{1, 2, 3, 4}},
	{0, 1, Shape{1, 2, 3, 4}},
	{0, 2, Shape{2, 1, 3, 4}},
	{0, 3, Shape{2, 3, 1, 4}},
	{0, 4, Shape{2, 3, 4, 1}},

	{1, 0, Shape{2, 1, 3, 4}},
	{1, 1, Shape{1, 2, 3, 4}},
	{1, 2, Shape{1, 2, 3, 4}},
	{1, 3, Shape{1, 3, 2, 4}},
	{1, 4, Shape{1, 3, 4, 2}},

	{2, 0, Shape{3, 1, 2, 4}},
	{2, 1, Shape{1, 3, 2, 4}},
	{2, 2, Shape{1, 2, 3, 4}},
	{2, 3, Shape{1, 2, 3, 4}},
	{2, 4, Shape{1, 2, 4, 3}},

	{3, 0, Shape{4, 1, 2, 3}},
	{3, 1, Shape{1, 4, 2, 3}},
	{3, 2, Shape{1, 2, 4, 3}},
	{3, 3, Shape{1, 2, 3, 4}},
	{3, 4, Shape{1, 2, 3, 4}},
}

// The RollAxis tests are directly adapted from Numpy's test cases.
func TestDense_RollAxis(t *testing.T) {
	assert := assert.New(t)
	var T *Dense
	var err error

	for _, rats := range rollaxisTests {
		T = New(Of(Byte), WithShape(1, 2, 3, 4))
		if _, err = T.RollAxis(rats.axis, rats.start, false); assert.NoError(err) {
			assert.True(rats.correctShape.Eq(T.Shape()), "%d %d Expected %v, got %v", rats.axis, rats.start, rats.correctShape, T.Shape())
		}
	}
}

var concatTests = []struct {
	name  string
	dt    Dtype
	a     interface{}
	shape Shape
	axis  int

	correctShape Shape
	correctData  interface{}
}{
	// Float64
	{"vector", Float64, nil, Shape{2}, 0, Shape{4}, []float64{0, 1, 0, 1}},
	{"matrix; axis 0 ", Float64, nil, Shape{2, 2}, 0, Shape{4, 2}, []float64{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Float64, nil, Shape{2, 2}, 1, Shape{2, 4}, []float64{0, 1, 0, 1, 2, 3, 2, 3}},

	// Float32
	{"vector", Float32, nil, Shape{2}, 0, Shape{4}, []float32{0, 1, 0, 1}},
	{"matrix; axis 0 ", Float32, nil, Shape{2, 2}, 0, Shape{4, 2}, []float32{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Float32, nil, Shape{2, 2}, 1, Shape{2, 4}, []float32{0, 1, 0, 1, 2, 3, 2, 3}},

	// Int
	{"vector", Int, nil, Shape{2}, 0, Shape{4}, []int{0, 1, 0, 1}},
	{"matrix; axis 0 ", Int, nil, Shape{2, 2}, 0, Shape{4, 2}, []int{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Int, nil, Shape{2, 2}, 1, Shape{2, 4}, []int{0, 1, 0, 1, 2, 3, 2, 3}},

	// Int64
	{"vector", Int64, nil, Shape{2}, 0, Shape{4}, []int64{0, 1, 0, 1}},
	{"matrix; axis 0 ", Int64, nil, Shape{2, 2}, 0, Shape{4, 2}, []int64{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Int64, nil, Shape{2, 2}, 1, Shape{2, 4}, []int64{0, 1, 0, 1, 2, 3, 2, 3}},

	// Int32
	{"vector", Int32, nil, Shape{2}, 0, Shape{4}, []int32{0, 1, 0, 1}},
	{"matrix; axis 0 ", Int32, nil, Shape{2, 2}, 0, Shape{4, 2}, []int32{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Int32, nil, Shape{2, 2}, 1, Shape{2, 4}, []int32{0, 1, 0, 1, 2, 3, 2, 3}},

	// Byte
	{"vector", Byte, nil, Shape{2}, 0, Shape{4}, []byte{0, 1, 0, 1}},
	{"matrix; axis 0 ", Byte, nil, Shape{2, 2}, 0, Shape{4, 2}, []byte{0, 1, 2, 3, 0, 1, 2, 3}},
	{"matrix; axis 1 ", Byte, nil, Shape{2, 2}, 1, Shape{2, 4}, []byte{0, 1, 0, 1, 2, 3, 2, 3}},

	// Bool
	{"vector", Bool, []bool{true, false}, Shape{2}, 0, Shape{4}, []bool{true, false, true, false}},
	{"matrix; axis 0 ", Bool, []bool{true, false, true, false}, Shape{2, 2}, 0, Shape{4, 2}, []bool{true, false, true, false, true, false, true, false}},
	{"matrix; axis 1 ", Bool, []bool{true, false, true, false}, Shape{2, 2}, 1, Shape{2, 4}, []bool{true, false, true, false, true, false, true, false}},
}

func TestDense_Concat(t *testing.T) {
	assert := assert.New(t)

	for _, cts := range concatTests {
		var T0, T1 *Dense

		if cts.a == nil {
			T0 = New(WithShape(cts.shape...), WithBacking(Range(cts.dt, 0, cts.shape.TotalSize())))
			T1 = New(WithShape(cts.shape...), WithBacking(Range(cts.dt, 0, cts.shape.TotalSize())))
		} else {
			T0 = New(WithShape(cts.shape...), WithBacking(cts.a))
			T1 = New(WithShape(cts.shape...), WithBacking(cloneArray(cts.a)))
		}

		T2, err := T0.Concat(cts.axis, T1)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(cts.correctShape.Eq(T2.Shape()))
		assert.Equal(cts.correctData, T2.Data())
	}
}

var simpleStackTests = []struct {
	name       string
	shape      Shape
	axis       int
	stackCount int

	correctShape Shape
	correctData  []float64
}{
	{"vector, axis 0, stack 2", Shape{2}, 0, 2, Shape{2, 2}, []float64{0, 1, 100, 101}},
	{"vector, axis 1, stack 2", Shape{2}, 1, 2, Shape{2, 2}, []float64{0, 100, 1, 101}},

	{"matrix, axis 0, stack 2", Shape{2, 3}, 0, 2, Shape{2, 2, 3}, []float64{0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105}},
	{"matrix, axis 1, stack 2", Shape{2, 3}, 1, 2, Shape{2, 2, 3}, []float64{0, 1, 2, 100, 101, 102, 3, 4, 5, 103, 104, 105}},
	{"matrix, axis 2, stack 2", Shape{2, 3}, 2, 2, Shape{2, 3, 2}, []float64{0, 100, 1, 101, 2, 102, 3, 103, 4, 104, 5, 105}},
	{"matrix, axis 0, stack 3", Shape{2, 3}, 0, 3, Shape{3, 2, 3}, []float64{0, 1, 2, 3, 4, 5, 100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205}},
	{"matrix, axis 1, stack 3", Shape{2, 3}, 1, 3, Shape{2, 3, 3}, []float64{0, 1, 2, 100, 101, 102, 200, 201, 202, 3, 4, 5, 103, 104, 105, 203, 204, 205}},
	{"matrix, axis 2, stack 3", Shape{2, 3}, 2, 3, Shape{2, 3, 3}, []float64{0, 100, 200, 1, 101, 201, 2, 102, 202, 3, 103, 203, 4, 104, 204, 5, 105, 205}},
}

var viewStackTests = []struct {
	name       string
	shape      Shape
	transform  []int
	slices     []Slice
	axis       int
	stackCount int

	correctShape Shape
	correctData  []float64
}{
	{"matrix(4x4)[1:3, 1:3] axis 0", Shape{4, 4}, nil, []Slice{makeRS(1, 3), makeRS(1, 3)}, 0, 2, Shape{2, 2, 2}, []float64{5, 6, 9, 10, 105, 106, 109, 110}},
	{"matrix(4x4)[1:3, 1:3] axis 1", Shape{4, 4}, nil, []Slice{makeRS(1, 3), makeRS(1, 3)}, 1, 2, Shape{2, 2, 2}, []float64{5, 6, 105, 106, 9, 10, 109, 110}},
	{"matrix(4x4)[1:3, 1:3] axis 2", Shape{4, 4}, nil, []Slice{makeRS(1, 3), makeRS(1, 3)}, 2, 2, Shape{2, 2, 2}, []float64{5, 105, 6, 106, 9, 109, 10, 110}},
}

func TestDense_Stack(t *testing.T) {
	assert := assert.New(t)
	var err error
	for _, sts := range simpleStackTests {
		T := New(WithShape(sts.shape...), WithBacking(Range(Float64, 0, sts.shape.TotalSize())))

		var stacked []*Dense
		for i := 0; i < sts.stackCount-1; i++ {
			offset := (i + 1) * 100
			T1 := New(WithShape(sts.shape...), WithBacking(Range(Float64, offset, sts.shape.TotalSize()+offset)))
			stacked = append(stacked, T1)
		}

		T2, err := T.Stack(sts.axis, stacked...)
		if err != nil {
			t.Error(err)
			continue
		}
		assert.True(sts.correctShape.Eq(T2.Shape()))
		assert.Equal(sts.correctData, T2.Data())
	}

	for _, sts := range viewStackTests {
		T := New(WithShape(sts.shape...), WithBacking(Range(Float64, 0, sts.shape.TotalSize())))
		switch {
		case sts.slices != nil && sts.transform == nil:
			var sliced Tensor
			if sliced, err = T.Slice(sts.slices...); err != nil {
				t.Error(err)
				continue
			}
			T = sliced.(*Dense)
		case sts.transform != nil && sts.slices == nil:
			T.T(sts.transform...)
		}

		var stacked []*Dense
		for i := 0; i < sts.stackCount-1; i++ {
			offset := (i + 1) * 100
			T1 := New(WithShape(sts.shape...), WithBacking(Range(Float64, offset, sts.shape.TotalSize()+offset)))
			switch {
			case sts.slices != nil && sts.transform == nil:
				var sliced Tensor
				if sliced, err = T1.Slice(sts.slices...); err != nil {
					t.Error(err)
					continue
				}
				T1 = sliced.(*Dense)
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
		assert.Equal(sts.correctData, T2.Data())
	}
}

/*
var denseApplyTests = []struct {
	a     interface{}
	reuse interface{}
	incr  interface{}
	shape Shape
	slice []Slice

	correct     interface{}
	correctIncr interface{}
}{
	// Float64
	{Range(Float64, 0, 6), Range(Float64, 52, 58), []float64{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]float64{-0, -1, -2, -3, -4, -5}, []float64{100, 99, 98, 97, 96, 95}},
	{Range(Float64, 0, 6), Range(Float64, 52, 55), []float64{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]float64{0, -1, -2}, []float64{100, 99, 98}},
	{Range(Float64, 0, 6), Range(Float64, 52, 54), []float64{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]float64{-1, -4}, []float64{99, 96}},

	// Float32
	{Range(Float32, 0, 6), Range(Float32, 52, 58), []float32{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]float32{-0, -1, -2, -3, -4, -5}, []float32{100, 99, 98, 97, 96, 95}},
	{Range(Float32, 0, 6), Range(Float32, 52, 55), []float32{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]float32{0, -1, -2}, []float32{100, 99, 98}},
	{Range(Float32, 0, 6), Range(Float32, 52, 54), []float32{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]float32{-1, -4}, []float32{99, 96}},

	// Int
	{Range(Int, 0, 6), Range(Int, 52, 58), []int{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]int{-0, -1, -2, -3, -4, -5}, []int{100, 99, 98, 97, 96, 95}},
	{Range(Int, 0, 6), Range(Int, 52, 55), []int{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]int{0, -1, -2}, []int{100, 99, 98}},
	{Range(Int, 0, 6), Range(Int, 52, 54), []int{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]int{-1, -4}, []int{99, 96}},

	// Int64
	{Range(Int64, 0, 6), Range(Int64, 52, 58), []int64{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]int64{-0, -1, -2, -3, -4, -5}, []int64{100, 99, 98, 97, 96, 95}},
	{Range(Int64, 0, 6), Range(Int64, 52, 55), []int64{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]int64{0, -1, -2}, []int64{100, 99, 98}},
	{Range(Int64, 0, 6), Range(Int64, 52, 54), []int64{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]int64{-1, -4}, []int64{99, 96}},

	// Int32
	{Range(Int32, 0, 6), Range(Int32, 52, 58), []int32{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]int32{-0, -1, -2, -3, -4, -5}, []int32{100, 99, 98, 97, 96, 95}},
	{Range(Int32, 0, 6), Range(Int32, 52, 55), []int32{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]int32{0, -1, -2}, []int32{100, 99, 98}},
	{Range(Int32, 0, 6), Range(Int32, 52, 54), []int32{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]int32{-1, -4}, []int32{99, 96}},

	// Byte
	{Range(Byte, 0, 6), Range(Byte, 52, 58), []byte{100, 100, 100, 100, 100, 100},
		Shape{2, 3}, nil,
		[]byte{0, 255, 254, 253, 252, 251}, []byte{100, 99, 98, 97, 96, 95}},
	{Range(Byte, 0, 6), Range(Byte, 52, 55), []byte{100, 100, 100},
		Shape{2, 3}, []Slice{ss(0)},
		[]byte{0, 255, 254}, []byte{100, 99, 98}},
	{Range(Byte, 0, 6), Range(Byte, 52, 54), []byte{100, 100},
		Shape{2, 3}, []Slice{nil, ss(1)},
		[]byte{255, 252}, []byte{99, 96}},
}

func TestDense_Apply(t *testing.T) {
	assert := assert.New(t)
	for i, dat := range denseApplyTests {
		T := New(WithBacking(dat.a), WithShape(dat.shape...))
		var err error
		if dat.slice != nil {
			if T, err = T.Slice(dat.slice...); err != nil {
				t.Errorf("Slicing failed %+v", err)
				continue
			}
		}

		var T2 Tensor
		var fn interface{}
		// safe
		switch T.t {
		case Float64:
			fn = invFloat64
		case Float32:
			fn = invFloat32
		case Int:
			fn = invInt
		case Int64:
			fn = invInt64
		case Int32:
			fn = invInt32
		case Byte:
			fn = invByte
		case Bool:
			fn = invBool
		}

		T2, err = T.Apply(fn)

		if err != nil {
			t.Errorf("Test %d errored: %+v", i, err)
			continue
		}

		assert.NotEqual(T2, T, "Test %d", i)
		assert.Equal(dat.correct, T2.Data(), "Test %d", i)

		// incr
		incr := New(WithBacking(dat.incr))
		T2, err = T.Apply(fn, WithIncr(incr))
		if err != nil {
			t.Errorf("Test %d errored: %+v", i, err)
			continue
		}

		assert.NotEqual(T2, T, "Test Incr %d", i)
		assert.Equal(dat.correctIncr, T2.Data(), "Test %d", i)
	}
}
*/
