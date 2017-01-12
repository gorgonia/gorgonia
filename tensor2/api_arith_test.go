package tensor

import (
	"testing"

	"github.com/chewxy/vecf64"
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

func TestDot(t *testing.T) {
	assert := assert.New(t)
	var a, b, c, r Tensor
	var A, B, R, R2 Tensor
	var s, s2 Tensor
	var incr Tensor
	var err error
	var expectedShape Shape
	var expectedData []float64

	// vector-vector
	t.Log("Vec⋅Vec")
	a = New(Of(Float64), WithShape(3, 1), WithBacking(Range(Float64, 0, 3)))
	b = New(Of(Float64), WithShape(3, 1), WithBacking(Range(Float64, 0, 3)))
	r, err = Dot(a, b)

	expectedShape = Shape{1}
	expectedData = []float64{5}
	assert.Nil(err)
	assert.Equal(expectedData, r.Data())
	assert.True(ScalarShape().Eq(r.Shape()))

	// vector-mat (which is the same as matᵀ*vec)
	t.Log("Vec⋅Mat dot, should be equal to Aᵀb")
	A = New(Of(Float64), WithShape(3, 2), WithBacking(Range(Float64, 0, 6)))
	R, err = Dot(b, A)

	expectedShape = Shape{2}
	expectedData = []float64{10, 13}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// mat-mat
	t.Log("Mat⋅Mat")
	A = New(Of(Float64), WithShape(4, 5), WithBacking(Range(Float64, 0, 20)))
	B = New(Of(Float64), WithShape(5, 10), WithBacking(Range(Float64, 2, 52)))
	R, err = Dot(A, B)
	expectedShape = Shape{4, 10}
	expectedData = []float64{
		320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 870,
		905, 940, 975, 1010, 1045, 1080, 1115, 1150, 1185, 1420, 1480,
		1540, 1600, 1660, 1720, 1780, 1840, 1900, 1960, 1970, 2055, 2140,
		2225, 2310, 2395, 2480, 2565, 2650, 2735,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// T-T
	t.Log("3T⋅3T")
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	B = New(Of(Float64), WithShape(3, 4, 2), WithBacking(Range(Float64, 0, 24)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 3, 3, 2}
	expectedData = []float64{
		28, 34,
		76, 82,
		124, 130,

		76, 98,
		252, 274,
		428, 450,

		124, 162,
		428, 466,
		732, 770,

		//

		172, 226,
		604, 658,
		1036, 1090,

		220, 290,
		780, 850,
		1340, 1410,

		268, 354,
		956, 1042,
		1644, 1730,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// T-T
	t.Log("3T⋅4T")
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	B = New(Of(Float64), WithShape(2, 3, 4, 5), WithBacking(Range(Float64, 0, 120)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 3, 2, 3, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94, 190, 196, 202, 208, 214, 310,
		316, 322, 328, 334, 430, 436, 442, 448, 454, 550, 556,
		562, 568, 574, 670, 676, 682, 688, 694, 190, 212, 234,
		256, 278, 630, 652, 674, 696, 718, 1070, 1092, 1114, 1136,
		1158, 1510, 1532, 1554, 1576, 1598, 1950, 1972, 1994, 2016, 2038,
		2390, 2412, 2434, 2456, 2478, 310, 348, 386, 424, 462, 1070,
		1108, 1146, 1184, 1222, 1830, 1868, 1906, 1944, 1982, 2590, 2628,
		2666, 2704, 2742, 3350, 3388, 3426, 3464, 3502, 4110, 4148, 4186,
		4224, 4262, 430, 484, 538, 592, 646, 1510, 1564, 1618, 1672,
		1726, 2590, 2644, 2698, 2752, 2806, 3670, 3724, 3778, 3832, 3886,
		4750, 4804, 4858, 4912, 4966, 5830, 5884, 5938, 5992, 6046, 550,
		620, 690, 760, 830, 1950, 2020, 2090, 2160, 2230, 3350, 3420,
		3490, 3560, 3630, 4750, 4820, 4890, 4960, 5030, 6150, 6220, 6290,
		6360, 6430, 7550, 7620, 7690, 7760, 7830, 670, 756, 842, 928,
		1014, 2390, 2476, 2562, 2648, 2734, 4110, 4196, 4282, 4368, 4454,
		5830, 5916, 6002, 6088, 6174, 7550, 7636, 7722, 7808, 7894, 9270,
		9356, 9442, 9528, 9614,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// T-v
	t.Log("3T⋅Vec")
	b = New(Of(Float64), WithShape(4), WithBacking(Range(Float64, 0, 4)))
	R, err = Dot(A, b)
	expectedShape = Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// v-T
	t.Log("Vec⋅3T")
	R2, err = Dot(b, B)
	expectedShape = Shape{2, 3, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94,
		190, 196, 202, 208, 214,
		310, 316, 322, 328, 334,

		430, 436, 442, 448, 454,
		550, 556, 562, 568, 574,
		670, 676, 682, 688, 694,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R2.Data())
	assert.Equal(expectedShape, R2.Shape())

	// m-3T
	t.Log("Mat⋅3T")
	A = New(Of(Float64), WithShape(2, 4), WithBacking(Range(Float64, 0, 8)))
	B = New(Of(Float64), WithShape(2, 4, 5), WithBacking(Range(Float64, 0, 40)))
	R, err = Dot(A, B)
	expectedShape = Shape{2, 2, 5}
	expectedData = []float64{
		70, 76, 82, 88, 94,
		190, 196, 202, 208, 214,

		190, 212, 234, 256, 278,
		630, 652, 674, 696, 718,
	}
	assert.Nil(err)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// test reuse

	// m-v with reuse
	t.Log("Mat⋅Vec with reuse")
	R = New(Of(Float64), WithShape(2))
	R2, err = Dot(A, b, WithReuse(R))
	expectedShape = Shape{2}
	expectedData = []float64{14, 38}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// 3T-vec with reuse
	t.Logf("3T⋅vec with reuse")
	R.Zero()
	A = New(Of(Float64), WithShape(2, 3, 4), WithBacking(Range(Float64, 0, 24)))
	R2, err = Dot(A, b, WithReuse(R))
	expectedShape = Shape{2, 3}
	expectedData = []float64{
		14, 38, 62,
		86, 110, 134,
	}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R2.Data())
	assert.Equal(expectedShape, R2.Shape())

	// v-m
	t.Log("vec⋅Mat with reuse")
	R = New(Of(Float64), WithShape(2))
	a = New(Of(Float64), WithShape(4), WithBacking(Range(Float64, 0, 4)))
	B = New(Of(Float64), WithShape(4, 2), WithBacking(Range(Float64, 0, 8)))
	R2, err = Dot(a, B, WithReuse(R))
	expectedShape = Shape{2}
	expectedData = []float64{28, 34}
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// test incr
	incrBack := make([]float64, 2)
	copy(incrBack, expectedData)
	incr = New(Of(Float64), WithBacking(incrBack), WithShape(2))
	R, err = Dot(a, B, WithIncr(incr))
	vecf64.Scale(expectedData, 2)
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(expectedData, R.Data())
	assert.Equal(expectedShape, R.Shape())

	// The Nearly Stupids

	s = New(Of(Float64), FromScalar(5.0))
	s2 = New(Of(Float64), FromScalar(10.0))
	R, err = Dot(s, s2)
	assert.Nil(err)
	assert.True(R.IsScalar())
	assert.Equal([]float64{50}, R.Data())

	R.Zero()
	R2, err = Dot(s, s2, WithReuse(R))
	assert.Nil(err)
	assert.True(R2.IsScalar())
	assert.Equal([]float64{50}, R2.Data())

	R, err = Dot(s, A)
	expectedData = vecf64.Range(0, 24)
	vecf64.Scale(expectedData, 5)
	assert.Nil(err)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())

	R.Zero()
	R2, err = Dot(s, A, WithReuse(R))
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(A.Shape(), R2.Shape())
	assert.Equal(expectedData, R2.Data())

	R, err = Dot(A, s)
	assert.Nil(err)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())

	R.Zero()
	R2, err = Dot(A, s, WithReuse(R))
	assert.Nil(err)
	assert.Equal(R, R2)
	assert.Equal(A.Shape(), R2.Shape())
	assert.Equal(expectedData, R2.Data())

	incr = New(Of(Float64), WithShape(R2.Shape()...))
	copy(incr.Data().([]float64), expectedData)
	incr2 := incr.Clone().(*Dense) // backup a copy for the following test
	vecf64.Scale(expectedData, 2)
	R, err = Dot(A, s, WithIncr(incr))
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())

	incr = incr2
	R, err = Dot(s, A, WithIncr(incr))
	assert.Nil(err)
	assert.Equal(incr, R)
	assert.Equal(A.Shape(), R.Shape())
	assert.Equal(expectedData, R.Data())

	incr = New(Of(Float64), FromScalar(float64(50)))
	R, err = Dot(s, s2, WithIncr(incr))
	assert.Nil(err)
	assert.True(R.IsScalar())
	assert.Equal([]float64{100}, R.Data())

	/* HERE BE STUPIDS */

	// different sizes of vectors
	c = New(Of(Float64), WithShape(1, 100))
	_, err = Dot(a, c)
	assert.NotNil(err)

	// vector mat, but with shape mismatch
	B = New(Of(Float64), WithShape(2, 3), WithBacking(Range(Float64, 0, 6)))
	_, err = Dot(b, B)
	assert.NotNil(err)

	// mat-mat but wrong reuse size
	A = New(Of(Float64), WithShape(2, 2))
	R = New(Of(Float64), WithShape(5, 10))
	_, err = Dot(A, B, WithReuse(R))
	assert.NotNil(err)

	// mat-vec but wrong reuse size
	b = New(Of(Float64), WithShape(2))
	_, err = Dot(A, b, WithReuse(R))
	assert.NotNil(err)

	// T-T but misaligned shape
	A = New(Of(Float64), WithShape(2, 3, 4))
	B = New(Of(Float64), WithShape(4, 2, 3))
	_, err = Dot(A, B)
	assert.NotNil(err)

}
