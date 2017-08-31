package tensor

import (
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/execution"
	"github.com/pkg/errors"
)

// StdEng is the default execution engine that comes with the tensors. To use other execution engines, use the WithEngine construction option.
type StdEng struct {
	execution.E
}

// makeArray allocates a slice for the array
func (e StdEng) makeArray(arr *array, t Dtype, size int) {
	memsize := calcMemSize(t, size)
	s := make([]byte, memsize)
	arr.t = t
	arr.L = size
	arr.C = size
	arr.Ptr = unsafe.Pointer(&s[0])
	arr.fix()
}

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
	case *array:
		switch st := src.(type) {
		case *array:
			copyArray(dt, st)
			return nil
		case arrayer:
			copyArray(dt, st.arrPtr())
			return nil
		}
	case arrayer:
		switch st := src.(type) {
		case *array:
			copyArray(dt.arrPtr(), st)
			return nil
		case arrayer:
			copyArray(dt.arrPtr(), st.arrPtr())
			return nil
		}
	}
	return errors.Errorf("Failed to copy %T %T", dst, src)
}

func (e StdEng) Accessible(mem Memory) (Memory, error) { return mem, nil }

func (e StdEng) WorksWith(order DataOrder) bool {
	if order.isColMajor() {
		return false
	}
	return true
}

func (e StdEng) checkAccessible(t Tensor) error {
	if !t.IsNativelyAccessible() {
		return errors.Errorf(inaccessibleData, t)
	}
	return nil
}
