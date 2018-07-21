// +build cuda

package gorgonia

import (
	"log"
	"os"
	"testing"

	"gorgonia.org/tensor"
)

func TestDevCUDA(t *testing.T) {
	// t.SkipNow()

	g := NewGraph()
	x := NewMatrix(g, Float64, WithShape(1024, 100), WithName("x"), WithInit(ValuesOf(2.0)))
	y := NewMatrix(g, Float64, WithShape(1024, 100), WithName("y"), WithInit(ValuesOf(8.0)))
	xpy := Must(Add(x, y))
	xmy := Must(Sub(x, y))
	xpy2 := Must(Square(xpy))
	WithName("xpy2")(xpy2)
	xmy2 := Must(Square(xmy))
	xpy2s := Must(Slice(xpy2, S(0)))

	logger := log.New(os.Stderr, "", 0)
	m := NewTapeMachine(g, WithLogger(logger), TraceExec(), WithWatchlist(), WithValueFmt("0x%x"))
	defer m.Close()

	prog, locMap, _ := Compile(g)
	t.Logf("prog:\n%v\n", prog)
	t.Logf("locMap %-v", FmtNodeMap(locMap))
	if err := m.RunAll(); err != nil {
		t.Errorf("%+v", err)
	}

	t.Logf("x: \n%v", x.Value())
	t.Logf("y: \n%v", y.Value())
	t.Logf("xpy \n%v", xpy.Value())
	t.Logf("xpy2: \n%v", xpy2.Value())
	t.Logf("xpy2s \n%v", xpy2s.Value())
	t.Logf("xmy2 \n%v", xmy2.Value())

	if assertGraphEngine(t, g, stdengType); t.Failed() {
		t.FailNow()
	}

}

func TestExternMetadata_Transfer(t *testing.T) {
	m := new(ExternMetadata)
	m.init([]int64{1024}) // allocate 1024 bytes

	v := tensor.New(tensor.Of(Float64), tensor.WithShape(2, 2))
	go func() {
		for s := range m.WorkAvailable() {
			m.DoWork()
			if s {
				m.syncChan <- struct{}{}
			}
		}
	}()

	//	 transfer from CPU to GPU
	v2, err := m.Transfer(Device(0), CPU, v, true)
	if err != nil {
		t.Error(err)
	}

	if vt, ok := v2.(*tensor.Dense); (ok && !vt.IsManuallyManaged()) || !ok {
		t.Errorf("Expected manually managed value")
	}
	t.Logf("v2: 0x%x", v2.Uintptr())

	// transfer from GPU to CPU
	v3, err := m.Transfer(CPU, Device(0), v2, true)
	if err != nil {
		t.Error(err)
	}
	if vt, ok := v3.(*tensor.Dense); (ok && vt.IsManuallyManaged()) || !ok {
		t.Errorf("Expected Go managed value")
	}
	t.Logf("v3: 0x%x", v3.Uintptr())

	// transfer from CPU to CPU
	v4, err := m.Transfer(CPU, CPU, v3, true)
	if err != nil {
		t.Error(err)
	}
	if v4 != v3 {
		t.Errorf("Expected the values to be returned exactly the same")
	}
}

func BenchmarkOneMilCUDA(b *testing.B) {
	xT := tensor.New(tensor.WithShape(1000000), tensor.WithBacking(tensor.Random(tensor.Float32, 1000000)))
	g := NewGraph()
	x := NewVector(g, Float32, WithShape(1000000), WithName("x"), WithValue(xT))
	Must(Sigmoid(x))

	m := NewTapeMachine(g)
	defer m.Close()

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
	defer m.Close()

	for n := 0; n < b.N; n++ {
		if err := m.RunAll(); err != nil {
			b.Fatalf("Failed at n: %d. Error: %v", n, err)
			break
		}
		m.Reset()
	}
}
