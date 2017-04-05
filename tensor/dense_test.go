package tensor

import (
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestDense_shallowClone(t *testing.T) {
	T := New(Of(Float64), WithBacking([]float64{1, 2, 3, 4}))
	T2 := T.shallowClone()
	T2.slice(0, 2)
	T2.float64s()[0] = 1000

	assert.Equal(t, T.Data().([]float64)[0:2], T2.Data())
}

func TestFromScalar(t *testing.T) {
	T := New(FromScalar(3.14))
	data := T.float64s()
	assert.Equal(t, []float64{3.14}, data)
}

func TestFromMemory(t *testing.T) {
	// dummy memory - this could be an externally malloc'd memory, or a mmap'ed file.
	// but here we're just gonna let Go manage memory.
	s := make([]float64, 100)
	ptr := uintptr(unsafe.Pointer(&s[0]))
	size := uintptr(100 * 8)

	T := New(Of(Float32), WithShape(50, 4), FromMemory(ptr, size))
	if len(T.float32s()) != 200 {
		t.Error("expected 200 float32s")
	}
	assert.Equal(t, make([]float32, 200), T.Data())

	fail := func() { New(FromMemory(ptr, size), Of(Float32)) }
	assert.Panics(t, fail, "Expected bad New() call to panic")
}

func Test_recycledDense(t *testing.T) {
	T := recycledDense(Float64, ScalarShape())
	assert.Equal(t, float64(0), T.Data())
}
