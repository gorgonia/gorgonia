package values

import (
	"fmt"

	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
)

// Size is a "tensor" representing the size of a dimension of a shape
type Size int

func (sz Size) Dtype() dtype.Dtype {
	return dtype.Datatype[Size]{}
}

func (sz Size) Shape() shapes.Shape { return shapes.ScalarShape() }

func (sz Size) Strides() []int { return nil }

func (sz Size) Dims() int { return 0 }

func (sz Size) Size() int { return 0 }

func (sz Size) Info() *tensor.AP { return nil }

func (sz Size) DataSize() int { return 1 }

func (sz Size) Data() []Size { return []Size{sz} }

func (sz Size) At(coords ...int) (Size, error) { return sz, errors.Errorf("Cannot do .At() on Size") }

func (sz Size) Engine() tensor.Engine { return nil }

func (sz Size) DataOrder() tensor.DataOrder { return 0 }
func (sz Size) Flags() tensor.MemoryFlag    { return 0 }
func (sz Size) IsNativelyAccessible() bool  { return true }
func (sz Size) IsManuallyManaged() bool     { return false }
func (sz Size) IsMaterializable() bool      { return false }
func (sz Size) RequiresIterator() bool      { return false }
func (sz Size) Iterator() tensor.Iterator   { return nil }
func (sz Size) MemSize() uintptr            { panic("NYI") }
func (sz Size) Uintptr() uintptr            { panic("NYI") }

func (sz Size) Reshape(...int) error              { return errors.NoOp{} }
func (sz Size) Restore()                          {}
func (sz Size) SetDataOrder(ord tensor.DataOrder) {}

func (sz Size) Unsqueeze(_ int) error { return errors.NoOp{} }

func (sz Size) Zero() {}

func (sz Size) Format(f fmt.State, c rune) {
	fmt.Fprintf(f, "%d", int(sz))
}

// required by Basic[DT]

func (s Size) AlikeAsDescWithStorage(opts ...tensor.ConsOpt) tensor.DescWithStorage { return s }

func (s Size) AlikeAsType(dt dtype.Dtype, opts ...tensor.ConsOpt) tensor.DescWithStorage {
	panic("NYI")
}

func (sz Size) AlikeAsBasic(opts ...tensor.ConsOpt) tensor.Basic[Size] { return sz }

func (sz Size) CloneAsBasic() tensor.Basic[Size] { return sz }

func (sz Size) SetAt(v Size, coord ...int) error { return errors.New("Cannot set scalar values") }
func (sz Size) Memset(v Size) error              { return errors.New("Cannot memset scalar") }
