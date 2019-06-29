package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/tensor"
)

func BenchmarkReshape_Dense(b *testing.B) {
	for _, rst := range reshapeTests {
		b.Run(rst.testName, func(b *testing.B) {
			g := NewGraph()
			tT := tensor.New(tensor.Of(tensor.Float64), tensor.WithShape(rst.input.Clone()...))
			T := NodeFromAny(g, tT)
			for i := 0; i < b.N; i++ {
				T2, err := Reshape(T, rst.to.Clone())
				switch {
				case rst.err && err == nil:
					b.Fatalf("Expected Error when testing %v", rst)
				case rst.err:
					continue
				case err != nil:
					b.Fatal(err)
				default:
					assert.True(b, rst.output.Eq(T2.Shape()), "expected both to be the same")
				}
			}
			m := NewTapeMachine(g)
			if err := m.RunAll(); err != nil {
				b.Fatal(err)
			}

		})
	}
}
