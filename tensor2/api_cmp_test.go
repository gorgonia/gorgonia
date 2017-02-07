package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* elEq */

var elEqTests = []struct {
	a         interface{}
	b         interface{}
	reuse     *Dense
	reuseSame *Dense

	correct     []bool
	correctSame interface{}
}{

	// Float64
	{a: float64(4),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           float64(4),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []float64{1, 1, 1, 1, 0, 0, 0},
	},

	// Float32
	{a: float32(4),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           float32(4),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []float32{1, 1, 1, 1, 0, 0, 0},
	},

	// Int
	{a: int(4),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           int(4),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int{1, 1, 1, 1, 0, 0, 0},
	},

	// Int64
	{a: int64(4),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           int64(4),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int64{1, 1, 1, 1, 0, 0, 0},
	},

	// Int32
	{a: int32(4),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           int32(4),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int32{1, 1, 1, 1, 0, 0, 0},
	},

	// Byte
	{a: byte(4),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []byte{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           byte(4),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []byte{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []byte{1, 1, 1, 1, 0, 0, 0},
	},
}

func TestElEq(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range elEqTests {
		// safe and not same
		T, err := ElEq(ats.a, ats.b)
		if err != nil {
			t.Errorf("Safe Test of ElEq %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest ElEq %d", i)

		// safe and same
		T, err = ElEq(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of ElEq %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test ElEq %d", i)

		// reuse and not same
		T, err = ElEq(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("Reuse Not Same Test of ElEq %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest ElEq %d", i)

		// reuse and same
		T, err = ElEq(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil {
			t.Errorf("Reuse Same Test of ElEq %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test ElEq %d", i)

		// unsafe and same
		T, err = ElEq(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("Unsafe Test of ElEq %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test ElEq %d", i)
	}

}

/* gt */

var gtTests = []struct {
	a         interface{}
	b         interface{}
	reuse     *Dense
	reuseSame *Dense

	correct     []bool
	correctSame interface{}
}{

	// Float64
	{a: float64(4),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []float64{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           float64(4),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float64{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float64{0, 0, 0, 0, 0, 0, 0},
	},

	// Float32
	{a: float32(4),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []float32{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           float32(4),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float32{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float32{0, 0, 0, 0, 0, 0, 0},
	},

	// Int
	{a: int(4),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           int(4),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int{0, 0, 0, 0, 0, 0, 0},
	},

	// Int64
	{a: int64(4),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int64{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           int64(4),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int64{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int64{0, 0, 0, 0, 0, 0, 0},
	},

	// Int32
	{a: int32(4),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int32{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           int32(4),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int32{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int32{0, 0, 0, 0, 0, 0, 0},
	},

	// Byte
	{a: byte(4),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []byte{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           byte(4),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []byte{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []byte{0, 0, 0, 0, 0, 0, 0},
	},
}

func TestGt(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range gtTests {
		// safe and not same
		T, err := Gt(ats.a, ats.b)
		if err != nil {
			t.Errorf("Safe Test of Gt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest Gt %d", i)

		// safe and same
		T, err = Gt(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of Gt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test Gt %d", i)

		// reuse and not same
		T, err = Gt(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("Reuse Not Same Test of Gt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest Gt %d", i)

		// reuse and same
		T, err = Gt(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil {
			t.Errorf("Reuse Same Test of Gt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test Gt %d", i)

		// unsafe and same
		T, err = Gt(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("Unsafe Test of Gt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test Gt %d", i)
	}

}

/* gte */

var gteTests = []struct {
	a         interface{}
	b         interface{}
	reuse     *Dense
	reuseSame *Dense

	correct     []bool
	correctSame interface{}
}{

	// Float64
	{a: float64(4),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float64{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           float64(4),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []float64{1, 1, 1, 1, 0, 0, 0},
	},

	// Float32
	{a: float32(4),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float32{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           float32(4),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []float32{1, 1, 1, 1, 0, 0, 0},
	},

	// Int
	{a: int(4),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           int(4),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int{1, 1, 1, 1, 0, 0, 0},
	},

	// Int64
	{a: int64(4),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int64{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           int64(4),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int64{1, 1, 1, 1, 0, 0, 0},
	},

	// Int32
	{a: int32(4),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int32{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           int32(4),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []int32{1, 1, 1, 1, 0, 0, 0},
	},

	// Byte
	{a: byte(4),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []byte{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           byte(4),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []byte{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, true, false, false, false},
		correctSame: []byte{1, 1, 1, 1, 0, 0, 0},
	},
}

func TestGte(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range gteTests {
		// safe and not same
		T, err := Gte(ats.a, ats.b)
		if err != nil {
			t.Errorf("Safe Test of Gte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest Gte %d", i)

		// safe and same
		T, err = Gte(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of Gte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test Gte %d", i)

		// reuse and not same
		T, err = Gte(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("Reuse Not Same Test of Gte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest Gte %d", i)

		// reuse and same
		T, err = Gte(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil {
			t.Errorf("Reuse Same Test of Gte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test Gte %d", i)

		// unsafe and same
		T, err = Gte(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("Unsafe Test of Gte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test Gte %d", i)
	}

}

/* lt */

var ltTests = []struct {
	a         interface{}
	b         interface{}
	reuse     *Dense
	reuseSame *Dense

	correct     []bool
	correctSame interface{}
}{

	// Float64
	{a: float64(4),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float64{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           float64(4),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []float64{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []float64{0, 0, 0, 0, 1, 1, 1},
	},

	// Float32
	{a: float32(4),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []float32{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           float32(4),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []float32{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []float32{0, 0, 0, 0, 1, 1, 1},
	},

	// Int
	{a: int(4),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           int(4),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []int{0, 0, 0, 0, 1, 1, 1},
	},

	// Int64
	{a: int64(4),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int64{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           int64(4),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int64{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []int64{0, 0, 0, 0, 1, 1, 1},
	},

	// Int32
	{a: int32(4),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []int32{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           int32(4),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []int32{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []int32{0, 0, 0, 0, 1, 1, 1},
	},

	// Byte
	{a: byte(4),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, false, false, false, false},
		correctSame: []byte{0, 0, 0, 0, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           byte(4),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, false, true, true, true},
		correctSame: []byte{1, 1, 1, 0, 1, 1, 1},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, false, true, true, true},
		correctSame: []byte{0, 0, 0, 0, 1, 1, 1},
	},
}

func TestLt(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range ltTests {
		// safe and not same
		T, err := Lt(ats.a, ats.b)
		if err != nil {
			t.Errorf("Safe Test of Lt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest Lt %d", i)

		// safe and same
		T, err = Lt(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of Lt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test Lt %d", i)

		// reuse and not same
		T, err = Lt(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("Reuse Not Same Test of Lt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest Lt %d", i)

		// reuse and same
		T, err = Lt(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil {
			t.Errorf("Reuse Same Test of Lt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test Lt %d", i)

		// unsafe and same
		T, err = Lt(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("Unsafe Test of Lt %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test Lt %d", i)
	}

}

/* lte */

var lteTests = []struct {
	a         interface{}
	b         interface{}
	reuse     *Dense
	reuseSame *Dense

	correct     []bool
	correctSame interface{}
}{

	// Float64
	{a: float64(4),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           float64(4),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float64{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float64), WithBacking([]float64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float64), WithBacking(make([]float64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float64{1, 1, 1, 1, 1, 1, 1},
	},

	// Float32
	{a: float32(4),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []float32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           float32(4),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float32{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Float32), WithBacking([]float32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Float32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Float32), WithBacking(make([]float32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []float32{1, 1, 1, 1, 1, 1, 1},
	},

	// Int
	{a: int(4),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           int(4),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int), WithBacking([]int{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int), WithBacking(make([]int, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int{1, 1, 1, 1, 1, 1, 1},
	},

	// Int64
	{a: int64(4),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int64{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           int64(4),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int64{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int64), WithBacking([]int64{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int64), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int64), WithBacking(make([]int64, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int64{1, 1, 1, 1, 1, 1, 1},
	},

	// Int32
	{a: int32(4),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []int32{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           int32(4),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int32{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Int32), WithBacking([]int32{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Int32), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Int32), WithBacking(make([]int32, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []int32{1, 1, 1, 1, 1, 1, 1},
	},

	// Byte
	{a: byte(4),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{false, false, false, true, false, false, false},
		correctSame: []byte{0, 0, 0, 1, 0, 0, 0},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           byte(4),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []byte{1, 1, 1, 1, 1, 1, 1},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 3, 2, 1})),
		b:           New(Of(Byte), WithBacking([]byte{1, 2, 3, 4, 5, 6, 7})),
		reuse:       New(Of(Byte), WithBacking(make([]bool, 7))),
		reuseSame:   New(Of(Byte), WithBacking(make([]byte, 7))),
		correct:     []bool{true, true, true, true, true, true, true},
		correctSame: []byte{1, 1, 1, 1, 1, 1, 1},
	},
}

func TestLte(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range lteTests {
		// safe and not same
		T, err := Lte(ats.a, ats.b)
		if err != nil {
			t.Errorf("Safe Test of Lte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "SafeTest Lte %d", i)

		// safe and same
		T, err = Lte(ats.a, ats.b, AsSameType())
		if err != nil {
			t.Errorf("Same Test of Lte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "SameType Test Lte %d", i)

		// reuse and not same
		T, err = Lte(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("Reuse Not Same Test of Lte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correct, T.Data(), "Reuse Not SameTest Lte %d", i)

		// reuse and same
		T, err = Lte(ats.a, ats.b, WithReuse(ats.reuseSame), AsSameType())
		if err != nil {
			t.Errorf("Reuse Same Test of Lte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Reuse Same Test Lte %d", i)

		// unsafe and same
		T, err = Lte(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("Unsafe Test of Lte %d errored: %+v", i, err)
			continue
		}
		assert.Equal(ats.correctSame, T.Data(), "Unsafe Test Lte %d", i)
	}

}
