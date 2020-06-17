package exprgraph

import (
	"fmt"
	"unsafe"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var _ gorgonia.Tensor = &Symbolic{}

// Symbolic is a representation of a symbolic tensor - it has no data
type Symbolic struct {
	tensor.AP
	dt tensor.Dtype
	g  *Graph
}

func NewSymbolic(g *Graph, dt tensor.Dtype, shape tensor.Shape) *Symbolic {
	strides := shape.CalcStrides()
	ap := tensor.MakeAP(shape, strides, 0, 0)
	return &Symbolic{AP: ap, dt: dt, g: g}
}

// DataSize returns the amount of data stored or accessible within the *Symbolic type. It's 0.
func (t *Symbolic) DataSize() int { return 0 }

// Dtype returns the associated dtype.
func (t *Symbolic) Dtype() tensor.Dtype { return t.dt }

// Engine returns the associated engine (always a *Graph)
func (t *Symbolic) Engine() tensor.Engine { return t.g }

// IsNativelyAccessible will always return false (there is no data).
func (t *Symbolic) IsNativelyAccessible() bool { return false }

// IsManuallyManaged will always return true. The Symbolic tensor does not have any data. So there is nothing for Go to manage.
func (t *Symbolic) IsManuallyManaged() bool { return true }

// MemSize will always return 0.
func (t *Symbolic) MemSize() uintptr { return 0 }

// Uintptr will always return 0.
func (t *Symbolic) Uintptr() uintptr { return 0 }

// Pointer will always return nil.
func (t *Symbolic) Pointer() unsafe.Pointer { return nil }

// ScalarValue will always return nil. (There is no data)
func (t *Symbolic) ScalarValue() interface{} { return nil }

func (t *Symbolic) Format(f fmt.State, c rune) { fmt.Fprintf(f, t.g.NameOf(t)) }

func (t *Symbolic) Data() interface{} { return nil }
