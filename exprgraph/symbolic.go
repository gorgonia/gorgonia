package exprgraph

/*
var _ Tensor = &header{}

// header is a representation of a header of tensor - it has no data
type header struct {
	tensor.AP
	dt tensor.Dtype
	g  *Graph
}

// newHeader tensor
func newHeader(g *Graph, dt tensor.Dtype, shape tensor.Shape) *header {
	strides := tensor.CalcStrides(shape)
	ap := tensor.MakeAP(shape, strides, 0, 0)
	return &header{AP: ap, dt: dt, g: g}
}

// DataSize returns the amount of data stored or accessible within the *Symbolic type. It's 0.
func (t *header) DataSize() int { return 0 }

// Dtype returns the associated dtype.
func (t *header) Dtype() tensor.Dtype { return t.dt }

// Engine returns the associated engine (always a *Graph)
func (t *header) Engine() tensor.Engine { return t.g }

// IsNativelyAccessible will always return false (there is no data).
func (t *header) IsNativelyAccessible() bool { return false }

// IsManuallyManaged will always return true. The Symbolic tensor does not have any data. So there is nothing for Go to manage.
func (t *header) IsManuallyManaged() bool { return true }

// MemSize will always return 0.
func (t *header) MemSize() uintptr { return 0 }

// Uintptr will always return 0.
func (t *header) Uintptr() uintptr { return 0 }

// Pointer will always return nil.
func (t *header) Pointer() unsafe.Pointer { return nil }

// ScalarValue will always return nil. (There is no data)
func (t *header) ScalarValue() interface{} { return nil }

// Format ...
func (t *header) Format(f fmt.State, c rune) {
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
func (t *header) Data() interface{} { return nil }

// Type returns the type of the *header. This implements hm.Typer.
func (t *header) Type() hm.Type {
	if t.Shape().IsScalar() {
		return t.dt
	}
	return types.TensorType{Dims: t.Shape().Dims(), Of: t.dt}
}
*/
