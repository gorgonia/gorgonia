package tensor

import "testing"

func BenchmarkNativeSet(b *testing.B) {
	a := makeArray(Float64, 10000)
	f := a.(f64s)
	for i := 0; i < b.N; i++ {
		for j := range f {
			f[j] = float64(j + 1)
		}
	}
}

func BenchmarkSetMethod(b *testing.B) {
	a := makeArray(Float64, 10000)
	for i := 0; i < b.N; i++ {
		for j := 0; j < a.Len(); j++ {
			a.Set(j, float64(j+1))
		}
	}
}
