package tensor

import "testing"

func BenchmarkDense_Transpose(b *testing.B) {
	T := New(WithShape(100, 100, 2), WithBacking(Range(Byte, 0, 100*100*2)))
	for i := 0; i < b.N; i++ {
		T.T()
		T.Transpose()
	}
}

func BenchmarkNativeSet(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	data := T.Data().([]float64)
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			data[next] = float64(next + 1)
		}
	}
}

func BenchmarkSetMethod(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			T.Set(next, float64(next+1))
		}
	}
}

func BenchmarkNativeGet(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	data := T.Data().([]float64)
	var f float64
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			f = data[next]
		}
	}
	_ = f
}

func BenchmarkGetMethod(b *testing.B) {
	T := New(WithShape(10000), Of(Float64))
	var f float64
	for i := 0; i < b.N; i++ {
		for next := 0; next < 10000; next++ {
			f = T.Get(next).(float64)
		}
	}
	_ = f
}

func BenchmarkGetWithIterator(b *testing.B) {
	T := New(WithShape(100, 100), Of(Float64))
	var f float64
	data := T.Data().([]float64)
	for i := 0; i < b.N; i++ {
		it := IteratorFromDense(T)
		var next int
		var err error
		for next, err = it.Start(); err == nil; next, err = it.Next() {
			f = data[next]
		}
		if _, ok := err.(NoOpError); !ok {
			b.Error("Error: %v", err)
		}
	}
	_ = f
}
