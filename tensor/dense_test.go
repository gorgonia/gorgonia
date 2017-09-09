package tensor

import (
	"math/rand"
	"testing"
	"testing/quick"
	"time"
	"unsafe"

	"github.com/stretchr/testify/assert"
)

func TestDense_shallowClone(t *testing.T) {
	T := New(Of(Float64), WithBacking([]float64{1, 2, 3, 4}))
	T2 := T.shallowClone()
	T2.slice(0, 2)
	T2.Float64s()[0] = 1000

	assert.Equal(t, T.Data().([]float64)[0:2], T2.Data())
}

func TestDense_Clone(t *testing.T) {
	assert := assert.New(t)
	cloneChk := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		if !q.Shape().Eq(a.Shape()) {
			t.Errorf("Shape Difference: %v %v", q.Shape(), a.Shape())
			return false
		}
		if len(q.Strides()) != len(a.Strides()) {
			t.Errorf("Stride Difference: %v %v", q.Strides(), a.Strides())
			return false
		}
		for i, s := range q.Strides() {
			if a.Strides()[i] != s {
				t.Errorf("Stride Difference: %v %v", q.Strides(), a.Strides())
				return false
			}
		}
		if q.o != a.o {
			t.Errorf("Data Order difference : %v %v", q.o, a.o)
			return false
		}

		if q.Δ != a.Δ {
			t.Errorf("Triangle Difference: %v  %v", q.Δ, a.Δ)
			return false
		}
		if q.flag != a.flag {
			t.Errorf("Flag difference : %v %v", q.flag, a.flag)
			return false
		}
		if q.e != a.e {
			t.Errorf("Engine difference; %T %T", q.e, a.e)
			return false
		}
		if q.oe != a.oe {
			t.Errorf("Optimized Engine difference; %T %T", q.oe, a.oe)
			return false
		}

		if len(q.transposeWith) != len(a.transposeWith) {
			t.Errorf("TransposeWith difference: %v %v", q.transposeWith, a.transposeWith)
			return false
		}

		assert.Equal(q.mask, a.mask, "mask difference")
		assert.Equal(q.maskIsSoft, a.maskIsSoft, "mask is soft ")
		return true
	}
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(cloneChk, &quick.Config{Rand: r}); err != nil {
		t.Error(err)
	}
}

func TestDenseMasked(t *testing.T) {
	T := New(Of(Float64), WithShape(3, 2))
	T.ResetMask()
	assert.Equal(t, []bool{false, false, false, false, false, false}, T.mask)

}

func TestFromScalar(t *testing.T) {
	T := New(FromScalar(3.14))
	data := T.Float64s()
	assert.Equal(t, []float64{3.14}, data)
}

func TestFromMemory(t *testing.T) {
	// dummy memory - this could be an externally malloc'd memory, or a mmap'ed file.
	// but here we're just gonna let Go manage memory.
	s := make([]float64, 100)
	ptr := uintptr(unsafe.Pointer(&s[0]))
	size := uintptr(100 * 8)

	T := New(Of(Float32), WithShape(50, 4), FromMemory(ptr, size))
	if len(T.Float32s()) != 200 {
		t.Error("expected 200 Float32s")
	}
	assert.Equal(t, make([]float32, 200), T.Data())
	assert.True(t, T.IsManuallyManaged(), "Unamanged %v |%v | q: %v", ManuallyManaged, T.flag, (T.flag>>ManuallyManaged)&MemoryFlag(1))

	fail := func() { New(FromMemory(ptr, size), Of(Float32)) }
	assert.Panics(t, fail, "Expected bad New() call to panic")
}

func Test_recycledDense(t *testing.T) {
	T := recycledDense(Float64, ScalarShape())
	assert.Equal(t, float64(0), T.Data())
	assert.Equal(t, StdEng{}, T.e)
	assert.Equal(t, StdEng{}, T.oe)
}
