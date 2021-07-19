package exprgraph

import (
	"fmt"
	"unsafe"

	"gorgonia.org/tensor"
)

var _ Tensor = &Symbolic{}

// Symbolic is a representation of a Symbolic tensor - it has no data
type Symbolic struct {
	tensor.AP
	e  tensor.Engine
	dt tensor.Dtype
	g  *Graph
}

// NewSymbolic tensor
func NewSymbolic(g *Graph, e tensor.Engine, dt tensor.Dtype, shape tensor.Shape) *Symbolic {
	strides := tensor.CalcStrides(shape)
	ap := tensor.MakeAP(shape, strides, 0, 0)
	return &Symbolic{AP: ap, dt: dt, g: g, e: e}
}

// DataSize returns the amount of data stored or accessible within the *Symbolic type. It's 0.
func (t *Symbolic) DataSize() int { return 0 }

// Dtype returns the associated dtype.
func (t *Symbolic) Dtype() tensor.Dtype { return t.dt }

// Engine returns the associated engine (always a *Graph)
func (t *Symbolic) Engine() tensor.Engine { return t.e }

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

// Format ...
func (t *Symbolic) Format(f fmt.State, c rune) {
	var name string
	if n := t.g.find(t); n != nil {
		name = n.name
	}

	switch {
	case f.Flag('#'), f.Flag('+'):
		fmt.Fprintf(f, "%v %v", name, t.Shape())
	default:
		fmt.Fprint(f, name)
	}
}

// Data returns nil because there's no associated data.
func (t *Symbolic) Data() interface{} { return nil }
