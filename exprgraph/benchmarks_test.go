package exprgraph

import (
	"fmt"
	"testing"

	"gorgonia.org/tensor"
)

func BenchmarkInsertion1(b *testing.B) {
	g := New(tensor.StdEng{})
	x := Make(g, "x", tensor.WithShape(), tensor.Of(tensor.Float64))

	for n := 0; n < b.N; n++ {
		g.Insert(x)
	}
}

func BenchmarkInsertion2(b *testing.B) {
	b.StopTimer()
	g := New(tensor.StdEng{})
	var list []Node
	for i := 0; i < 1025; i++ {
		node := Make(g, fmt.Sprintf("Node%d", i), tensor.WithShape(), tensor.Of(tensor.Float32))
		g.Insert(node)
		list = append(list, node)
	}
	b.ResetTimer()
	b.StartTimer()

	for n := 0; n < b.N; n++ {
		g.Insert(list[n%1025])
	}
}
