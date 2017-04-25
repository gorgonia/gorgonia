// +build cuda

package gorgonia

import (
	"log"
	"os"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func TestDevCUDA(t *testing.T) {
	t.SkipNow()

	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(1024, 100), WithName("x"), WithInit(ValuesOf(2.0)))
	y := NewMatrix(g, Float64, WithShape(1024, 100), WithName("y"), WithInit(ValuesOf(2.0)))
	xpy := Must(Add(x, y))
	xmy := Must(Sub(x, y))
	xpy2 := Must(Square(xpy))
	WithName("xpy2")(xpy2)
	xmy2 := Must(Square(xmy))
	xpy2s := Must(Slice(xpy2, S(0)))

	logger := log.New(os.Stderr, "", 0)
	m := NewTapeMachine(g, WithLogger(logger))

	prog, locMap, _ := Compile(g)
	t.Logf("prog:\n%v\n", prog)
	t.Logf("locMap %-v", FmtNodeMap(locMap))
	if err := m.RunAll(); err != nil {
		t.Errorf("%+v", err)
	}

	t.Logf("x: %v", x.Value())
	t.Logf("y: %v", y.Value())
	t.Logf("xpy %v", xpy.Value())
	t.Logf("xpy2: \n%v", xpy2.Value())
	t.Logf("xpy2s \n%v", xpy2s.Value())
	t.Logf("xmy2 \n%v", xmy2.Value())
}

func BenchmarkOneMilCUDA(b *testing.B) {
	xT := tensor.New(tensor.WithShape(1000000), tensor.WithBacking(tensor.Random(tensor.Float32, 1000000)))
	g := NewGraph()
	x := NewVector(g, Float32, WithShape(1000000), WithName("x"), WithValue(xT))
	Must(Sigmoid(x))

	m := NewTapeMachine(g)

	// runtime.LockOSThread()
	for n := 0; n < b.N; n++ {
		if err := m.RunAll(); err != nil {
			b.Fatalf("Failed at n: %d. Error: %v", n, err)
			break
		}
		m.Reset()
	}
	// runtime.UnlockOSThread()
}

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
}
