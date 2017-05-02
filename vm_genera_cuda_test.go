// +build cuda

package gorgonia

import (
	"log"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
)

func TestGeneraCUDA_init(t *testing.T) {
	g, x, y, z := simpleMatEqn()
	zs := Must(Slice(z, S(0))) // not a CUDA op (for now)
	ex := NewVector(g, Float64, WithName("extra"), WithShape(2))
	xs := Must(Slice(x, S(1)))
	zpe := Must(Add(zs, ex))
	zpepxpe := Must(Add(xs, zpe))
	Must(Sum(zpepxpe))

	xV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 1, 2, 3}))
	yV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{5, 4, 3, 2}))
	eV := tensor.New(tensor.WithShape(2), tensor.WithBacking([]float64{1000, 50000}))

	Let(x, xV)
	Let(y, yV)
	Let(ex, eV)

	// logger := log.New(os.Stderr, "", 0)
	// m := NewLispMachine(g, WithLogger(logger), WithWatchlist(), LogBothDir())
	m := NewLispMachine(g)

	t.Logf("%v", m.sorted)
	t.Logf("%v %v", m.cpumem, m.gpumem)
	t.Logf("%v", m.df.devTransChildren)
	t.Logf("%v", m.df.devTransRepl[m.sorted[0]])
	if err := m.RunAll(); err != nil {
		t.Errorf("Error %v", err)
	}

	for _, n := range m.sorted {
		if n.boundTo != nil {
			if dv, ok := n.boundTo.(*dualValue); ok {
				log.Printf("\tEncountered %v 0x%x | 0x%x", n, dv.Value.Uintptr(), dv.d.Uintptr())
			} else {
				log.Printf("\tEncountered %v 0x%x", n, n.boundTo.Uintptr())
			}
		}
	}

	var xG, yG Value
	var err error
	if xG, err = x.Grad(); err != nil {
		t.Fatal(err)
	}
	if yG, err = y.Grad(); err != nil {
		t.Fatal(err)
	}
	t.Logf("xG:\n%v", xG)
	t.Logf("yG:\n%v", yG)

	// Compile(g)

}

func TestGenera_ForceCPU(t *testing.T) {
	g, x, y, z := simpleMatEqn()

	xV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 1, 2, 3}))
	yV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{5, 4, 3, 2}))

	Let(x, xV)
	Let(y, yV)
	m := NewLispMachine(g, WithManualGradient())
	m.ForceCPU()

	if err := m.RunAll(); err != nil {
		t.Errorf("%v", err)
	}

	t.Logf("%v", z.Value())
}

func TestGenera_Backprop(t *testing.T) {
	g, x, y, z := simpleMatEqn()
	Must(Sum(z))

	xV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{0, 1, 2, 3}))
	yV := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float64{5, 4, 3, 2}))

	Let(x, xV)
	Let(y, yV)
	m := NewLispMachine(g)

	if err := m.RunAll(); err != nil {
		t.Errorf("%v", err)
	}
	t.Logf("x.Value: 0x%x | d: 0x%x", x.boundTo.(*dualValue).Value.Uintptr(), x.boundTo.(*dualValue).d.Uintptr())
}
