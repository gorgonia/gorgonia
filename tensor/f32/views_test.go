package tensorf32

import (
	"testing"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/stretchr/testify/assert"
)

// slice materialization
var materializeTests1 = []struct {
	s      types.Shape
	slices []types.Slice

	correct []float32
}{
	// vectors
	{types.Shape{5}, []types.Slice{nil}, []float32{0, 1, 2, 3, 4}},         // Vec[:]
	{types.Shape{5}, []types.Slice{ss(1)}, []float32{1}},                   // Vec[1]
	{types.Shape{5}, []types.Slice{makeRS(1, 3)}, []float32{1, 2}},         // Vec[1:3]
	{types.Shape{5, 1}, []types.Slice{nil}, []float32{0, 1, 2, 3, 4}},      // ColVec[:]
	{types.Shape{5, 1}, []types.Slice{makeRS(1, 3)}, []float32{1, 2}},      // ColVec[1:3]
	{types.Shape{1, 5}, []types.Slice{nil}, []float32{0, 1, 2, 3, 4}},      // RowVec[:]
	{types.Shape{1, 5}, []types.Slice{nil, makeRS(1, 3)}, []float32{1, 2}}, // RowVec[:, 1:3]

	// matrices
	{types.Shape{4, 4}, []types.Slice{nil}, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},      // Mat[:]
	{types.Shape{4, 4}, []types.Slice{nil, nil}, []float32{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}}, // Mat[:, :]
	{types.Shape{4, 4}, []types.Slice{ss(0), ss(1)}, []float32{1}},                                                // Mat[0, 1]
	{types.Shape{4, 4}, []types.Slice{nil, makeRS(0, 2)}, []float32{0, 1, 4, 5, 8, 9, 12, 13}},                    // Mat[:, 0:2]
	{types.Shape{4, 4}, []types.Slice{makeRS(0, 2)}, []float32{0, 1, 2, 3, 4, 5, 6, 7}},                           // Mat[0:2]
	{types.Shape{4, 4}, []types.Slice{makeRS(0, 2), makeRS(0, 2)}, []float32{0, 1, 4, 5}},                         // Mat[0:2, 0:2]

	// 3-Tensors
	{types.Shape{2, 3, 4}, []types.Slice{nil}, RangeFloat32(0, 24)},                                                               // T[:]
	{types.Shape{2, 3, 4}, []types.Slice{ss(0), nil}, RangeFloat32(0, 12)},                                                        // T[0, :]
	{types.Shape{2, 3, 4}, []types.Slice{ss(0), ss(0), nil}, RangeFloat32(0, 4)},                                                  // T[0, 0, :]
	{types.Shape{2, 3, 4}, []types.Slice{ss(0), ss(0), ss(1)}, []float32{1}},                                                      // T[0, 0, 1]
	{types.Shape{2, 3, 4}, []types.Slice{ss(0), makeRS(1, 3)}, RangeFloat32(4, 12)},                                               // T[0, 1:3]
	{types.Shape{2, 3, 4}, []types.Slice{nil, makeRS(1, 3)}, []float32{4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23}}, // T[:, 1:3]
}

// transpose
var materializeTests2 = []struct {
	s types.Shape
	t []int

	noop    bool
	correct []float32
}{
	// vectors
	{types.Shape{5}, []int{}, true, []float32{0, 1, 2, 3, 4}}, // vector  transpose is a no-op
	{types.Shape{5, 1}, []int{}, false, []float32{0, 1, 2, 3, 4}},
	{types.Shape{1, 5}, []int{}, false, []float32{0, 1, 2, 3, 4}},

	// matrices
	{types.Shape{4, 2}, []int{0, 1}, true, RangeFloat32(0, 8)},
	{types.Shape{4, 2}, []int{}, false, []float32{0, 2, 4, 6, 1, 3, 5, 7}},

	// 3T
	{types.Shape{2, 3, 4}, []int{0, 1, 2}, true, RangeFloat32(0, 24)},
	{types.Shape{2, 3, 4}, []int{1, 0, 2}, false, []float32{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}},
	{types.Shape{2, 3, 4}, []int{1, 2, 0}, false, []float32{0, 12, 1, 13, 2, 14, 3, 15, 4, 16, 5, 17, 6, 18, 7, 19, 8, 20, 9, 21, 10, 22, 11, 23}},
	{types.Shape{2, 3, 4}, []int{2, 1, 0}, false, []float32{0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23}},
	{types.Shape{2, 3, 4}, []int{0, 2, 1}, false, []float32{0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23}},
}

func TestMaterialize(t *testing.T) {
	assert := assert.New(t)

	var T, T2, _ *Tensor
	var err error

	for i, mts := range materializeTests1 {
		T = NewTensor(WithShape(mts.s...), WithBacking(RangeFloat32(0, mts.s.TotalSize())))
		if T2, err = T.Slice(mts.slices...); err != nil {
			t.Errorf("Slice Test %d: %v", i, err)
			continue
		}

		T3 = T2.Materialize().(*Tensor)
		assert.Equal(mts.correct, T3.data, "Slice Test %d", i)
		assert.Equal(T2.Shape(), T3.Shape(), "Slice Test %d", i)
		T.data[1] = 5000
		assert.Equal(mts.correct, T3.data, "Slice Test %d", i)
	}

	for i, mts := range materializeTests2 {
		T = NewTensor(WithShape(mts.s...), WithBacking(RangeFloat32(0, mts.s.TotalSize())))
		if err = T.T(mts.t...); err != nil {
			t.Errorf("Transpose Test %d: %v", i, err)
			continue
		}

		T2 = T.Materialize().(*Tensor)
		correctStrides := T.Shape().CalcStrides()
		assert.Equal(mts.correct, T2.data, "Transpose Test %d", i)
		assert.Equal(T.Shape(), T2.Shape(), "Transpose Test %d", i)
		assert.Equal(correctStrides, T2.Strides(), "Transpose Test %d", i)

		if !mts.noop {
			T.data[1] = 5000
			assert.Equal(mts.correct, T2.data, "Transpose Test %d", i)
		}
	}

}
