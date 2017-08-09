package tensor

import (
	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/pkg/errors"
)

// StdEng is the default execution engine that comes with the tensors. To use other execution engines, use the WithEngine construction option.
type StdEng struct {
	execution.E
}

func (e StdEng) makeArray(t Dtype, size int) array { return makeArray(t, size) }

func (e StdEng) AllocAccessible() bool             { return true }
func (e StdEng) Alloc(size int64) (Memory, error)  { return nil, noopError{} }
func (e StdEng) Free(mem Memory, size int64) error { return nil }
func (e StdEng) Memset(mem Memory, val interface{}) error {
	if ms, ok := mem.(MemSetter); ok {
		return ms.Memset(val)
	}
	return errors.Errorf("Cannot memset %v with StdEng")
}

func (e StdEng) Memclr(mem Memory) {
	if z, ok := mem.(Zeroer); ok {
		z.Zero()
	}
	return
}

func (e StdEng) Memcpy(dst, src Memory) error {
	switch dt := dst.(type) {
	case array:
		switch st := src.(type) {
		case array:
			copyArray(dt, st)
			return nil
		case arrayer:
			copyArray(dt, st.arr())
			return nil
		}
	case arrayer:
		switch st := src.(type) {
		case array:
			copyArray(dt.arr(), st)
			return nil
		case arrayer:
			copyArray(dt.arr(), st.arr())
			return nil
		}
	}
	return errors.Errorf("Failed to copy %T %T", dst, src)
}

func (e StdEng) Accessible(mem Memory) (Memory, error) { return mem, nil }

func (e StdEng) DataOrder() DataOrder { return DataOrder(0) }
