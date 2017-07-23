package tensor

import (
	"errors"
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
)

func checkErr(t *testing.T, expected bool, err error, name string, id interface{}) (cont bool) {
	switch {
	case expected:
		if err == nil {
			t.Errorf("Expected error in test %v (%v)", name, id)
		}
		return true
	case !expected && err != nil:
		t.Errorf("Test %v (%v) errored: %+v", name, id, err)
		return true
	}
	return false
}

func sliceApproxf64(a, b []float64, fn func(a, b float64) bool) bool {
	if len(a) != len(b) {
		return false
	}

	for i, v := range a {
		if math.IsNaN(v) {
			if !alikef64(v, b[i]) {
				return false
			}
		}
		if !fn(v, b[i]) {
			return false
		}
	}
	return true
}

func RandomFloat64(size int) []float64 {
	r := make([]float64, size)
	for i := range r {
		r[i] = rand.NormFloat64()
	}
	return r
}

// fakemem is a byteslice, while making it a Memory
type fakemem []byte

func (m fakemem) Uintptr() uintptr        { return uintptr(unsafe.Pointer(&m[0])) }
func (m fakemem) MemSize() uintptr        { return uintptr(len(m)) }
func (m fakemem) Pointer() unsafe.Pointer { return unsafe.Pointer(&m[0]) }

// dummyEngine implements Engine. The bool indicates whether the data is native-accessible
type dummyEngine bool

func (e dummyEngine) AllocAccessible() bool { return bool(e) }
func (e dummyEngine) Alloc(size int64) (Memory, error) {
	ps := make(fakemem, int(size))
	return ps, nil
}
func (e dummyEngine) Free(mem Memory, size int64) error        { return nil }
func (e dummyEngine) Memset(mem Memory, val interface{}) error { return nil }
func (e dummyEngine) Memclr(mem Memory)                        {}
func (e dummyEngine) Memcpy(dst, src Memory) error {
	if e {
		var a, b storage.Header
		a.Ptr = src.Pointer()
		a.L = int(src.MemSize())
		a.C = int(src.MemSize())

		b.Ptr = dst.Pointer()
		b.L = int(dst.MemSize())
		b.C = int(dst.MemSize())

		abs := *(*[]byte)(unsafe.Pointer(&a))
		bbs := *(*[]byte)(unsafe.Pointer(&b))
		copy(bbs, abs)
		return nil
	}
	return errors.New("Unable to copy ")
}
func (e dummyEngine) Accessible(mem Memory) (Memory, error) { return mem, nil }
func (e dummyEngine) DataOrder() DataOrder                  { return DataOrder(0) }
