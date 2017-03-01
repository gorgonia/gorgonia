// +build cuda

package gorgonia

import (
	"log"
	"runtime"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func TestExternMetadata_ElemGridSize(t *testing.T) {
	var m ExternMetadata
	m.init()
}

func TestDevCUDA(t *testing.T) {
	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(100, 200), WithName("x"))
	y := NewMatrix(g, Float64, WithShape(100, 200), WithName("y"))
	xpy := Must(Add(x, y))
	xmy := Must(Sub(x, y))
	xpy2 := Must(Square(xpy))
	xmy2 := Must(Square(xmy))
	xpy2s := Must(Slice(xpy2, S(0)))

	prog, locMap, _ := Compile(g)
	// m := NewTapeMachine(prog, locMap, UseCudaFor("add"))

	t.Logf("prog:\n%v\n", prog)
	t.Logf("locMap %v", locMap)
	// if err := m.RunAll(); err != nil {
	// 	t.Error(err)
	// }
	t.Logf("xpy2 %v", xpy2s.Value())
	t.Logf("xmy2 %v", xmy2.Value())
}

func BenchmarkOneMilCUDA(b *testing.B) {
	xT := tensor.New(tensor.WithShape(1000000), tensor.WithBacking(tensor.Random(tensor.Float32, 1000000)))
	g := NewGraph()
	x := NewVector(g, Float32, WithShape(1000000), WithName("x"), WithValue(xT))
	Must(Sigmoid(x))

	prog, locMap, _ := Compile(g)
	m := NewTapeMachine(prog, locMap, UseCudaFor())

	runtime.LockOSThread()
	for n := 0; n < b.N; n++ {
		if err := m.RunAll(); err != nil {
			log.Printf("Failed at n: %d. Error: %v", n, err)
			break
		}
		m.Reset()
	}
	runtime.UnlockOSThread()
}

func BenchmarkOneMil(b *testing.B) {
	xT := tensor.New(tensor.WithShape(1000000), tensor.WithBacking(tensor.Random(tensor.Float32, 1000000)))
	g := NewGraph()
	x := NewVector(g, Float32, WithShape(1000000), WithName("x"), WithValue(xT))
	Must(Sigmoid(x))

	prog, locMap, _ := Compile(g)
	m := NewTapeMachine(prog, locMap)

	runtime.LockOSThread()
	for n := 0; n < b.N; n++ {
		if err := m.RunAll(); err != nil {
			log.Printf("Failed at n: %d. Error: %v", n, err)
			break
		}
		m.Reset()
	}
	runtime.UnlockOSThread()
}
