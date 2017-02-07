package tensor

import "testing"

func BenchmarkDense_Transpose(b *testing.B) {
	T := New(WithShape(10000, 10000, 2), WithBacking(Range(Byte, 0, 10000*10000*2)))
	for i := 0; i < b.N; i++ {
		T.T()
		T.Transpose()
	}
}
