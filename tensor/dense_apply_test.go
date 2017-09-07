package tensor

import (
	"math/rand"
	"testing"
	"testing/quick"
	"time"
	"unsafe"
)

func getMutateVal(dt Dtype) interface{} {
	switch dt {
	case Int:
		return int(1)
	case Int8:
		return int8(1)
	case Int16:
		return int16(1)
	case Int32:
		return int32(1)
	case Int64:
		return int64(1)
	case Uint:
		return uint(1)
	case Uint8:
		return uint8(1)
	case Uint16:
		return uint16(1)
	case Uint32:
		return uint32(1)
	case Uint64:
		return uint64(1)
	case Float32:
		return float32(1)
	case Float64:
		return float64(1)
	case Complex64:
		var c complex64 = 1
		return c
	case Complex128:
		var c complex128 = 1
		return c
	case Bool:
		return true
	case String:
		return "Hello World"
	case Uintptr:
		return uintptr(0xdeadbeef)
	case UnsafePointer:
		return unsafe.Pointer(uintptr(0xdeadbeef))
	}
	return nil
}

func getMutateFn(dt Dtype) interface{} {
	switch dt {
	case Int:
		return mutateI
	case Int8:
		return mutateI8
	case Int16:
		return mutateI16
	case Int32:
		return mutateI32
	case Int64:
		return mutateI64
	case Uint:
		return mutateU
	case Uint8:
		return mutateU8
	case Uint16:
		return mutateU16
	case Uint32:
		return mutateU32
	case Uint64:
		return mutateU64
	case Float32:
		return mutateF32
	case Float64:
		return mutateF64
	case Complex64:
		return mutateC64
	case Complex128:
		return mutateC128
	case Bool:
		return mutateB
	case String:
		return mutateStr
	case Uintptr:
		return mutateUintptr
	case UnsafePointer:
		return mutateUnsafePointer
	}
	return nil
}

func TestDense_Apply(t *testing.T) {
	var r *rand.Rand
	mut := func(q *Dense) bool {
		var mutVal interface{}
		if mutVal = getMutateVal(q.Dtype()); mutVal == nil {
			return true // we'll temporarily skip those we cannot mutate/get a mutation value
		}
		var fn interface{}
		if fn = getMutateFn(q.Dtype()); fn == nil {
			return true // we'll skip those that we cannot mutate
		}

		we, eqFail := willerr(q, nil, nil)
		_, ok := q.Engine().(Mapper)
		we = we || !ok

		a := q.Clone().(*Dense)
		correct := q.Clone().(*Dense)
		correct.Memset(mutVal)
		ret, err := a.Apply(fn)
		if err, retEarly := qcErrCheck(t, "Apply", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if !qcEqCheck(t, a.Dtype(), eqFail, correct.Data(), ret.Data()) {
			return false
		}

		// wrong fn type/illogical values
		if _, err = a.Apply(getMutateFn); err == nil {
			t.Error("Expected an error")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(mut, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Applying mutation function failed %v", err)
	}
}

func TestDense_Apply_unsafe(t *testing.T) {
	var r *rand.Rand
	mut := func(q *Dense) bool {
		var mutVal interface{}
		if mutVal = getMutateVal(q.Dtype()); mutVal == nil {
			return true // we'll temporarily skip those we cannot mutate/get a mutation value
		}
		var fn interface{}
		if fn = getMutateFn(q.Dtype()); fn == nil {
			return true // we'll skip those that we cannot mutate
		}

		we, eqFail := willerr(q, nil, nil)
		_, ok := q.Engine().(Mapper)
		we = we || !ok

		a := q.Clone().(*Dense)
		correct := q.Clone().(*Dense)
		correct.Memset(mutVal)
		ret, err := a.Apply(fn, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Apply", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if !qcEqCheck(t, a.Dtype(), eqFail, correct.Data(), ret.Data()) {
			return false
		}
		if ret != a {
			t.Error("Expected ret == correct (Unsafe option was used)")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(mut, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Applying mutation function failed %v", err)
	}
}

func TestDense_Apply_reuse(t *testing.T) {
	var r *rand.Rand
	mut := func(q *Dense) bool {
		var mutVal interface{}
		if mutVal = getMutateVal(q.Dtype()); mutVal == nil {
			return true // we'll temporarily skip those we cannot mutate/get a mutation value
		}
		var fn interface{}
		if fn = getMutateFn(q.Dtype()); fn == nil {
			return true // we'll skip those that we cannot mutate
		}

		we, eqFail := willerr(q, nil, nil)
		_, ok := q.Engine().(Mapper)
		we = we || !ok

		a := q.Clone().(*Dense)
		reuse := q.Clone().(*Dense)
		reuse.Zero()
		correct := q.Clone().(*Dense)
		correct.Memset(mutVal)
		ret, err := a.Apply(fn, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Apply", a, nil, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		if !qcEqCheck(t, a.Dtype(), eqFail, correct.Data(), ret.Data()) {
			return false
		}
		if ret != reuse {
			t.Error("Expected ret == correct (Unsafe option was used)")
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(mut, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Applying mutation function failed %v", err)
	}
}
