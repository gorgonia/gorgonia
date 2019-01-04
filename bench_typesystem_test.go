package gorgonia

import "testing"

func BenchmarkTypeSystem(b *testing.B) {
	g := NewGraph()
	x := g.NewTensor(Float64, 2, WithName("x"), WithShape(10, 10))
	y := g.NewTensor(Float64, 2, WithName("y"), WithShape(10, 10))
	op := newEBOByType(addOpType, Float64, Float64)
	for i := 0; i < b.N; i++ {
		inferNodeType(op, x, y)
	}
}
