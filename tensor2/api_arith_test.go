package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

/* add */

var addTests = []struct {
	a interface{}
	b interface{}

	reuse *Dense
	incr  *Dense

	correct0, correct1, correct2 interface{}
}{
	// Float64
	{a: float64(1),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{2, 3, 4, 5},
		correct1: []float64{102, 103, 104, 105},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        float64(1),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{2, 3, 4, 5},
		correct1: []float64{102, 103, 104, 105},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{2, 4, 6, 8},
		correct1: []float64{102, 104, 106, 108},
	},
	// Float32
	{a: float32(1),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{2, 3, 4, 5},
		correct1: []float32{102, 103, 104, 105},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        float32(1),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{2, 3, 4, 5},
		correct1: []float32{102, 103, 104, 105},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{2, 4, 6, 8},
		correct1: []float32{102, 104, 106, 108},
	},
	// Int
	{a: int(1),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{2, 3, 4, 5},
		correct1: []int{102, 103, 104, 105},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        int(1),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{2, 3, 4, 5},
		correct1: []int{102, 103, 104, 105},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{2, 4, 6, 8},
		correct1: []int{102, 104, 106, 108},
	},
	// Int64
	{a: int64(1),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{2, 3, 4, 5},
		correct1: []int64{102, 103, 104, 105},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        int64(1),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{2, 3, 4, 5},
		correct1: []int64{102, 103, 104, 105},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{2, 4, 6, 8},
		correct1: []int64{102, 104, 106, 108},
	},
	// Int32
	{a: int32(1),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{2, 3, 4, 5},
		correct1: []int32{102, 103, 104, 105},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        int32(1),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{2, 3, 4, 5},
		correct1: []int32{102, 103, 104, 105},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{2, 4, 6, 8},
		correct1: []int32{102, 104, 106, 108},
	},
	// Byte
	{a: byte(1),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{2, 3, 4, 5},
		correct1: []byte{102, 103, 104, 105},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        byte(1),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{2, 3, 4, 5},
		correct1: []byte{102, 103, 104, 105},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{2, 4, 6, 8},
		correct1: []byte{102, 104, 106, 108},
	},
}

func TestAdd(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range addTests {
		// safe
		T, err := Add(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test Add %d", i)

		// incr
		T, err = Add(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test Add %d", i)

		// reuse
		T, err = Add(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test Add %d", i)

		// unsafe
		T, err = Add(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test Add %d", i)
	}
}

/* sub */

var subTests = []struct {
	a interface{}
	b interface{}

	reuse *Dense
	incr  *Dense

	correct0, correct1, correct2 interface{}
}{
	// Float64
	{a: float64(1),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{0, -1, -2, -3},
		correct1: []float64{100, 99, 98, 97},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        float64(1),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{0, 1, 2, 3},
		correct1: []float64{100, 101, 102, 103},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{0, 0, 0, 0},
		correct1: []float64{100, 100, 100, 100},
	},
	// Float32
	{a: float32(1),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{0, -1, -2, -3},
		correct1: []float32{100, 99, 98, 97},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        float32(1),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{0, 1, 2, 3},
		correct1: []float32{100, 101, 102, 103},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{0, 0, 0, 0},
		correct1: []float32{100, 100, 100, 100},
	},
	// Int
	{a: int(1),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{0, -1, -2, -3},
		correct1: []int{100, 99, 98, 97},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        int(1),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{0, 1, 2, 3},
		correct1: []int{100, 101, 102, 103},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{0, 0, 0, 0},
		correct1: []int{100, 100, 100, 100},
	},
	// Int64
	{a: int64(1),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{0, -1, -2, -3},
		correct1: []int64{100, 99, 98, 97},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        int64(1),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{0, 1, 2, 3},
		correct1: []int64{100, 101, 102, 103},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{0, 0, 0, 0},
		correct1: []int64{100, 100, 100, 100},
	},
	// Int32
	{a: int32(1),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{0, -1, -2, -3},
		correct1: []int32{100, 99, 98, 97},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        int32(1),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{0, 1, 2, 3},
		correct1: []int32{100, 101, 102, 103},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{0, 0, 0, 0},
		correct1: []int32{100, 100, 100, 100},
	},
	// Byte
	{a: byte(1),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{0, 255, 254, 253},
		correct1: []byte{100, 99, 98, 97},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        byte(1),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{0, 1, 2, 3},
		correct1: []byte{100, 101, 102, 103},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{0, 0, 0, 0},
		correct1: []byte{100, 100, 100, 100},
	},
}

func TestSub(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range subTests {
		// safe
		T, err := Sub(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test Sub %d", i)

		// incr
		T, err = Sub(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test Sub %d", i)

		// reuse
		T, err = Sub(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test Sub %d", i)

		// unsafe
		T, err = Sub(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test Sub %d", i)
	}
}

/* mul */

var mulTests = []struct {
	a interface{}
	b interface{}

	reuse *Dense
	incr  *Dense

	correct0, correct1, correct2 interface{}
}{
	// Float64
	{a: float64(1),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 2, 3, 4},
		correct1: []float64{101, 102, 103, 104},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        float64(1),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 2, 3, 4},
		correct1: []float64{101, 102, 103, 104},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 4, 9, 16},
		correct1: []float64{101, 104, 109, 116},
	},
	// Float32
	{a: float32(1),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 2, 3, 4},
		correct1: []float32{101, 102, 103, 104},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        float32(1),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 2, 3, 4},
		correct1: []float32{101, 102, 103, 104},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 4, 9, 16},
		correct1: []float32{101, 104, 109, 116},
	},
	// Int
	{a: int(1),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 2, 3, 4},
		correct1: []int{101, 102, 103, 104},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        int(1),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 2, 3, 4},
		correct1: []int{101, 102, 103, 104},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 4, 9, 16},
		correct1: []int{101, 104, 109, 116},
	},
	// Int64
	{a: int64(1),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 2, 3, 4},
		correct1: []int64{101, 102, 103, 104},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        int64(1),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 2, 3, 4},
		correct1: []int64{101, 102, 103, 104},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 4, 9, 16},
		correct1: []int64{101, 104, 109, 116},
	},
	// Int32
	{a: int32(1),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 2, 3, 4},
		correct1: []int32{101, 102, 103, 104},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        int32(1),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 2, 3, 4},
		correct1: []int32{101, 102, 103, 104},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 4, 9, 16},
		correct1: []int32{101, 104, 109, 116},
	},
	// Byte
	{a: byte(1),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 2, 3, 4},
		correct1: []byte{101, 102, 103, 104},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        byte(1),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 2, 3, 4},
		correct1: []byte{101, 102, 103, 104},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 4, 9, 16},
		correct1: []byte{101, 104, 109, 116},
	},
}

func TestMul(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range mulTests {
		// safe
		T, err := Mul(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test Mul %d", i)

		// incr
		T, err = Mul(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test Mul %d", i)

		// reuse
		T, err = Mul(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test Mul %d", i)

		// unsafe
		T, err = Mul(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test Mul %d", i)
	}
}

/* div */

var divTests = []struct {
	a interface{}
	b interface{}

	reuse *Dense
	incr  *Dense

	correct0, correct1, correct2 interface{}
}{
	// Float64
	{a: float64(24),
		b:        New(Of(Float64), WithBacking([]float64{2, 4, 6, 8})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{12, 6, 4, 3},
		correct1: []float64{112, 106, 104, 103},
	},
	{a: New(Of(Float64), WithBacking([]float64{2, 4, 6, 8})),
		b:        float64(2),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 2, 3, 4},
		correct1: []float64{101, 102, 103, 104},
	},
	{a: New(Of(Float64), WithBacking([]float64{2, 4, 6, 8})),
		b:        New(Of(Float64), WithBacking([]float64{2, 4, 6, 8})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 1, 1, 1},
		correct1: []float64{101, 101, 101, 101},
	},
	// Float32
	{a: float32(24),
		b:        New(Of(Float32), WithBacking([]float32{2, 4, 6, 8})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{12, 6, 4, 3},
		correct1: []float32{112, 106, 104, 103},
	},
	{a: New(Of(Float32), WithBacking([]float32{2, 4, 6, 8})),
		b:        float32(2),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 2, 3, 4},
		correct1: []float32{101, 102, 103, 104},
	},
	{a: New(Of(Float32), WithBacking([]float32{2, 4, 6, 8})),
		b:        New(Of(Float32), WithBacking([]float32{2, 4, 6, 8})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 1, 1, 1},
		correct1: []float32{101, 101, 101, 101},
	},
	// Int
	{a: int(24),
		b:        New(Of(Int), WithBacking([]int{2, 4, 6, 8})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{12, 6, 4, 3},
		correct1: []int{112, 106, 104, 103},
	},
	{a: New(Of(Int), WithBacking([]int{2, 4, 6, 8})),
		b:        int(2),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 2, 3, 4},
		correct1: []int{101, 102, 103, 104},
	},
	{a: New(Of(Int), WithBacking([]int{2, 4, 6, 8})),
		b:        New(Of(Int), WithBacking([]int{2, 4, 6, 8})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 1, 1, 1},
		correct1: []int{101, 101, 101, 101},
	},
	// Int64
	{a: int64(24),
		b:        New(Of(Int64), WithBacking([]int64{2, 4, 6, 8})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{12, 6, 4, 3},
		correct1: []int64{112, 106, 104, 103},
	},
	{a: New(Of(Int64), WithBacking([]int64{2, 4, 6, 8})),
		b:        int64(2),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 2, 3, 4},
		correct1: []int64{101, 102, 103, 104},
	},
	{a: New(Of(Int64), WithBacking([]int64{2, 4, 6, 8})),
		b:        New(Of(Int64), WithBacking([]int64{2, 4, 6, 8})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 1, 1, 1},
		correct1: []int64{101, 101, 101, 101},
	},
	// Int32
	{a: int32(24),
		b:        New(Of(Int32), WithBacking([]int32{2, 4, 6, 8})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{12, 6, 4, 3},
		correct1: []int32{112, 106, 104, 103},
	},
	{a: New(Of(Int32), WithBacking([]int32{2, 4, 6, 8})),
		b:        int32(2),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 2, 3, 4},
		correct1: []int32{101, 102, 103, 104},
	},
	{a: New(Of(Int32), WithBacking([]int32{2, 4, 6, 8})),
		b:        New(Of(Int32), WithBacking([]int32{2, 4, 6, 8})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 1, 1, 1},
		correct1: []int32{101, 101, 101, 101},
	},
	// Byte
	{a: byte(24),
		b:        New(Of(Byte), WithBacking([]byte{2, 4, 6, 8})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{12, 6, 4, 3},
		correct1: []byte{112, 106, 104, 103},
	},
	{a: New(Of(Byte), WithBacking([]byte{2, 4, 6, 8})),
		b:        byte(2),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 2, 3, 4},
		correct1: []byte{101, 102, 103, 104},
	},
	{a: New(Of(Byte), WithBacking([]byte{2, 4, 6, 8})),
		b:        New(Of(Byte), WithBacking([]byte{2, 4, 6, 8})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 1, 1, 1},
		correct1: []byte{101, 101, 101, 101},
	},
}

func TestDiv(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range divTests {
		// safe
		T, err := Div(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test Div %d", i)

		// incr
		T, err = Div(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test Div %d", i)

		// reuse
		T, err = Div(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test Div %d", i)

		// unsafe
		T, err = Div(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test Div %d", i)
	}
}

/* pow */

var powTests = []struct {
	a interface{}
	b interface{}

	reuse *Dense
	incr  *Dense

	correct0, correct1, correct2 interface{}
}{
	// Float64
	{a: float64(1),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 1, 1, 1},
		correct1: []float64{101, 101, 101, 101},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        float64(1),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 2, 3, 4},
		correct1: []float64{101, 102, 103, 104},
	},
	{a: New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		b:        New(Of(Float64), WithBacking([]float64{1, 2, 3, 4})),
		incr:     New(Of(Float64), WithBacking([]float64{100, 100, 100, 100})),
		reuse:    New(Of(Float64), WithBacking([]float64{200, 200, 200, 200})),
		correct0: []float64{1, 4, 27, 256},
		correct1: []float64{101, 104, 127, 356},
	},
	// Float32
	{a: float32(1),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 1, 1, 1},
		correct1: []float32{101, 101, 101, 101},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        float32(1),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 2, 3, 4},
		correct1: []float32{101, 102, 103, 104},
	},
	{a: New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		b:        New(Of(Float32), WithBacking([]float32{1, 2, 3, 4})),
		incr:     New(Of(Float32), WithBacking([]float32{100, 100, 100, 100})),
		reuse:    New(Of(Float32), WithBacking([]float32{200, 200, 200, 200})),
		correct0: []float32{1, 4, 27, 256},
		correct1: []float32{101, 104, 127, 356},
	},
	// Int
	{a: int(1),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 1, 1, 1},
		correct1: []int{101, 101, 101, 101},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        int(1),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 2, 3, 4},
		correct1: []int{101, 102, 103, 104},
	},
	{a: New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		b:        New(Of(Int), WithBacking([]int{1, 2, 3, 4})),
		incr:     New(Of(Int), WithBacking([]int{100, 100, 100, 100})),
		reuse:    New(Of(Int), WithBacking([]int{200, 200, 200, 200})),
		correct0: []int{1, 4, 27, 256},
		correct1: []int{101, 104, 127, 356},
	},
	// Int64
	{a: int64(1),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 1, 1, 1},
		correct1: []int64{101, 101, 101, 101},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        int64(1),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 2, 3, 4},
		correct1: []int64{101, 102, 103, 104},
	},
	{a: New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		b:        New(Of(Int64), WithBacking([]int64{1, 2, 3, 4})),
		incr:     New(Of(Int64), WithBacking([]int64{100, 100, 100, 100})),
		reuse:    New(Of(Int64), WithBacking([]int64{200, 200, 200, 200})),
		correct0: []int64{1, 4, 27, 256},
		correct1: []int64{101, 104, 127, 356},
	},
	// Int32
	{a: int32(1),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 1, 1, 1},
		correct1: []int32{101, 101, 101, 101},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        int32(1),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 2, 3, 4},
		correct1: []int32{101, 102, 103, 104},
	},
	{a: New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		b:        New(Of(Int32), WithBacking([]int32{1, 2, 3, 4})),
		incr:     New(Of(Int32), WithBacking([]int32{100, 100, 100, 100})),
		reuse:    New(Of(Int32), WithBacking([]int32{200, 200, 200, 200})),
		correct0: []int32{1, 4, 27, 256},
		correct1: []int32{101, 104, 127, 356},
	},
	// Byte
	{a: byte(1),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 1, 1, 1},
		correct1: []byte{101, 101, 101, 101},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        byte(1),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 2, 3, 4},
		correct1: []byte{101, 102, 103, 104},
	},
	{a: New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		b:        New(Of(Byte), WithBacking([]byte{1, 2, 3, 4})),
		incr:     New(Of(Byte), WithBacking([]byte{100, 100, 100, 100})),
		reuse:    New(Of(Byte), WithBacking([]byte{200, 200, 200, 200})),
		correct0: []byte{1, 4, 27, 0},
		correct1: []byte{101, 104, 127, 100},
	},
}

func TestPow(t *testing.T) {
	assert := assert.New(t)
	for i, ats := range powTests {
		// safe
		T, err := Pow(ats.a, ats.b)
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Safe Test Pow %d", i)

		// incr
		T, err = Pow(ats.a, ats.b, WithIncr(ats.incr))
		if err != nil {
			t.Errorf("%+v", err)
		}
		assert.Equal(ats.correct1, T.Data(), "Incr Test Pow %d", i)

		// reuse
		T, err = Pow(ats.a, ats.b, WithReuse(ats.reuse))
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Reuse Test Pow %d", i)

		// unsafe
		T, err = Pow(ats.a, ats.b, UseUnsafe())
		if err != nil {
			t.Errorf("%v", err)
		}
		assert.Equal(ats.correct0, T.Data(), "Unsafe Test Pow %d", i)
	}
}
