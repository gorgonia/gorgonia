package tensor

import (
	"testing"

	"github.com/chewxy/vecf64"
)

func BenchmarkDense_Mul_Unsafe(b *testing.B) {
	A := New(WithShape(100, 100, 2), WithBacking(Range(Float64, 0, 100*100*2)))
	B := New(WithShape(100, 100, 2), WithBacking(Range(Float64, 1, (100*100*2)+1)))

	for i := 0; i < b.N; i++ {
		A.Mul(B, UseUnsafe())
	}
}

func BenchmarkNative_Mul_Unsafe(b *testing.B) {
	A := Range(Float64, 0, 100*100*2).([]float64)
	B := Range(Float64, 1, (100*100*2)+1).([]float64)

	f := func(a, b []float64) {
		for i, v := range a {
			a[i] = v * b[i]
		}
	}

	for i := 0; i < b.N; i++ {
		f(A, B)
	}
}

func BenchmarkNative_Mul_Unsafe_vec(b *testing.B) {
	A := Range(Float64, 0, 100*100*2).([]float64)
	B := Range(Float64, 1, (100*100*2)+1).([]float64)

	for i := 0; i < b.N; i++ {
		vecf64.Mul(A, B)
	}
}

func BenchmarkAPI_Mul_Unsafe(b *testing.B) {
	A := New(WithShape(100, 100, 2), WithBacking(Range(Float64, 0, 100*100*2)))
	B := New(WithShape(100, 100, 2), WithBacking(Range(Float64, 1, (100*100*2)+1)))

	for i := 0; i < b.N; i++ {
		Mul(A, B, UseUnsafe())
	}
}

func BenchmarkDense_ContiguousSliced_Mul_Unsafe(b *testing.B) {
	A := New(WithShape(4, 100, 100), WithBacking(Range(Float64, 0, 100*100*4)))
	B := New(WithShape(2, 100, 100), WithBacking(Range(Float64, 1, (100*100*2)+1)))
	Sliced, _ := A.Slice(makeRS(1, 3)) // result should be contiguous

	for i := 0; i < b.N; i++ {
		Mul(Sliced, B, UseUnsafe())
	}
}

func BenchmarkDense_NonContiguousSliced_Mul_Unsafe(b *testing.B) {
	A := New(WithShape(100, 4, 100), WithBacking(Range(Float64, 0, 100*100*4)))
	B := New(WithShape(100, 2, 100), WithBacking(Range(Float64, 1, (100*100*2)+1)))
	Sliced, _ := A.Slice(nil, makeRS(1, 3)) // result should be non-contiguous

	for i := 0; i < b.N; i++ {
		Mul(Sliced, B, UseUnsafe())
	}
}

func BenchmarkAPI_MulScalar_Unsafe(b *testing.B) {
	A := New(WithShape(100, 100, 2), WithBacking(Range(Float64, 0, 100*100*2)))
	B := 3.141
	for i := 0; i < b.N; i++ {
		Mul(A, B, UseUnsafe())
	}
}

func BenchmarkNative_MulScalar_Unsafe(b *testing.B) {
	A := Range(Float64, 0, 100*100*2).([]float64)
	B := 3.141
	f := func(a []float64, b float64) {
		for i := range a {
			a[i] *= b
		}
	}

	for i := 0; i < b.N; i++ {
		f(A, B)
	}
}
