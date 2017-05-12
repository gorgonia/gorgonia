// +build !cuda

package gorgonia

import (
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func BenchmarkOneMil(b *testing.B) {
	xT := tensor.New(tensor.WithShape(1000000), tensor.WithBacking(tensor.Random(tensor.Float32, 1000000)))
	g := NewGraph()
	x := NewVector(g, Float32, WithShape(1000000), WithName("x"), WithValue(xT))
	Must(Sigmoid(x))

	m := NewTapeMachine(g)

	for n := 0; n < b.N; n++ {
		if err := m.RunAll(); err != nil {
			b.Fatalf("Failed at n: %d. Error: %v", n, err)
			break
		}
		m.Reset()
	}
	runtime.GC()
}
