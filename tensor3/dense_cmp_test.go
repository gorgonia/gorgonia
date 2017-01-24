package tensor

import (
	"reflect"
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Eq */

func TestDense_eqDD_Transitivity(t *testing.T) {
	fB := func(a, b, c *QCDenseB) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for bool failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for bool failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for bool failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fB, nil); err != nil {
		t.Error(err)
	}
	fI := func(a, b, c *QCDenseI) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b, c *QCDenseI8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b, c *QCDenseI16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b, c *QCDenseI32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b, c *QCDenseI64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for int64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b, c *QCDenseU) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b, c *QCDenseU8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b, c *QCDenseU16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b, c *QCDenseU32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b, c *QCDenseU64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uint64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b, c *QCDenseUintptr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uintptr failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uintptr failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for uintptr failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b, c *QCDenseF32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b, c *QCDenseF64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for float64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fC64 := func(a, b, c *QCDenseC64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fC64, nil); err != nil {
		t.Error(err)
	}
	fC128 := func(a, b, c *QCDenseC128) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex128 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex128 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for complex128 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fC128, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b, c *QCDenseStr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for string failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for string failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for string failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
	fUnsafePointer := func(a, b, c *QCDenseUnsafePointer) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq transitivity for unsafe.Pointer failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for unsafe.Pointer failed (bxc): %v", err)
			return false
		}
		if axc, err = a.eqDD(c.Dense); err != nil {
			t.Errorf("Test Eq transitivity for unsafe.Pointer failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUnsafePointer, nil); err != nil {
		t.Error(err)
	}
}

func Test_Dense_eqDD_funcOpts(t *testing.T) {
	fB := func(a, b *QCDenseB) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for bool failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for bool failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fB, nil); err != nil {
		t.Error(err)
	}
	fI := func(a, b *QCDenseI) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for int failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for int failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for int failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b *QCDenseI8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int8, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for int8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b *QCDenseI16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int16, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for int16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b *QCDenseI32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int32, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for int32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b *QCDenseI64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int64, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for int64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b *QCDenseU) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for uint failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b *QCDenseU8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint8, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for uint8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b *QCDenseU16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint16, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for uint16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b *QCDenseU32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint32, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for uint32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b *QCDenseU64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint64, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for uint64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b *QCDenseUintptr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for uintptr failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for uintptr failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b *QCDenseF32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float32, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for float32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b *QCDenseF64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float64, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for float64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fC64 := func(a, b *QCDenseC64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for complex64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for complex64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Complex64, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for complex64 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for complex64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for complex64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fC64, nil); err != nil {
		t.Error(err)
	}
	fC128 := func(a, b *QCDenseC128) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for complex128 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for complex128 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Complex128, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Eq as same type reuse for complex128 failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq as same type reuse for complex128 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.eqDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Eq for complex128 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fC128, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b *QCDenseStr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for string failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for string failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
	fUnsafePointer := func(a, b *QCDenseUnsafePointer) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.eqDD(b.Dense); err != nil {
			t.Errorf("Test Eq reuse for unsafe.Pointer failed(axb): %v", err)
		}
		if ret, err = a.eqDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Eq reuse for unsafe.Pointer failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUnsafePointer, nil); err != nil {
		t.Error(err)
	}
}

/* Gt */

func TestDense_gtDD_Transitivity(t *testing.T) {
	fI := func(a, b, c *QCDenseI) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b, c *QCDenseI8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b, c *QCDenseI16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b, c *QCDenseI32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b, c *QCDenseI64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for int64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b, c *QCDenseU) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b, c *QCDenseU8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b, c *QCDenseU16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b, c *QCDenseU32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b, c *QCDenseU64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uint64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b, c *QCDenseUintptr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uintptr failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uintptr failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for uintptr failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b, c *QCDenseF32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b, c *QCDenseF64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for float64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b, c *QCDenseStr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt transitivity for string failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for string failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gtDD(c.Dense); err != nil {
			t.Errorf("Test Gt transitivity for string failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

func Test_Dense_gtDD_funcOpts(t *testing.T) {
	fI := func(a, b *QCDenseI) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for int failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for int failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for int failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b *QCDenseI8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int8, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for int8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b *QCDenseI16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int16, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for int16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b *QCDenseI32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int32, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for int32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b *QCDenseI64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int64, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for int64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b *QCDenseU) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for uint failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b *QCDenseU8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint8, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for uint8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b *QCDenseU16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint16, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for uint16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b *QCDenseU32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint32, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for uint32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b *QCDenseU64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint64, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for uint64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b *QCDenseUintptr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for uintptr failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for uintptr failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b *QCDenseF32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float32, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for float32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b *QCDenseF64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float64, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gt as same type reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt as same type reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gtDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gt for float64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b *QCDenseStr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gtDD(b.Dense); err != nil {
			t.Errorf("Test Gt reuse for string failed(axb): %v", err)
		}
		if ret, err = a.gtDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gt reuse for string failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

/* Gte */

func TestDense_gteDD_Transitivity(t *testing.T) {
	fI := func(a, b, c *QCDenseI) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b, c *QCDenseI8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b, c *QCDenseI16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b, c *QCDenseI32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b, c *QCDenseI64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for int64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b, c *QCDenseU) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b, c *QCDenseU8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b, c *QCDenseU16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b, c *QCDenseU32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b, c *QCDenseU64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uint64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b, c *QCDenseUintptr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uintptr failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uintptr failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for uintptr failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b, c *QCDenseF32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b, c *QCDenseF64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for float64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b, c *QCDenseStr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte transitivity for string failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for string failed (bxc): %v", err)
			return false
		}
		if axc, err = a.gteDD(c.Dense); err != nil {
			t.Errorf("Test Gte transitivity for string failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

func Test_Dense_gteDD_funcOpts(t *testing.T) {
	fI := func(a, b *QCDenseI) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for int failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for int failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for int failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b *QCDenseI8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int8, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for int8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b *QCDenseI16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int16, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for int16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b *QCDenseI32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int32, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for int32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b *QCDenseI64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int64, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for int64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b *QCDenseU) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for uint failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b *QCDenseU8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint8, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for uint8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b *QCDenseU16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint16, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for uint16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b *QCDenseU32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint32, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for uint32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b *QCDenseU64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint64, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for uint64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b *QCDenseUintptr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for uintptr failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for uintptr failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b *QCDenseF32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float32, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for float32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b *QCDenseF64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float64, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Gte as same type reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte as same type reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.gteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Gte for float64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b *QCDenseStr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.gteDD(b.Dense); err != nil {
			t.Errorf("Test Gte reuse for string failed(axb): %v", err)
		}
		if ret, err = a.gteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Gte reuse for string failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

/* Lt */

func TestDense_ltDD_Transitivity(t *testing.T) {
	fI := func(a, b, c *QCDenseI) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b, c *QCDenseI8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b, c *QCDenseI16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b, c *QCDenseI32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b, c *QCDenseI64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for int64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b, c *QCDenseU) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b, c *QCDenseU8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b, c *QCDenseU16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b, c *QCDenseU32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b, c *QCDenseU64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uint64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b, c *QCDenseUintptr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uintptr failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uintptr failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for uintptr failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b, c *QCDenseF32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b, c *QCDenseF64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for float64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b, c *QCDenseStr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt transitivity for string failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for string failed (bxc): %v", err)
			return false
		}
		if axc, err = a.ltDD(c.Dense); err != nil {
			t.Errorf("Test Lt transitivity for string failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

func Test_Dense_ltDD_funcOpts(t *testing.T) {
	fI := func(a, b *QCDenseI) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for int failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for int failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for int failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b *QCDenseI8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int8, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for int8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b *QCDenseI16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int16, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for int16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b *QCDenseI32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int32, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for int32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b *QCDenseI64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int64, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for int64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b *QCDenseU) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for uint failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b *QCDenseU8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint8, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for uint8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b *QCDenseU16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint16, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for uint16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b *QCDenseU32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint32, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for uint32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b *QCDenseU64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint64, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for uint64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b *QCDenseUintptr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for uintptr failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for uintptr failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b *QCDenseF32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float32, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for float32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b *QCDenseF64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float64, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lt as same type reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt as same type reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.ltDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lt for float64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b *QCDenseStr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.ltDD(b.Dense); err != nil {
			t.Errorf("Test Lt reuse for string failed(axb): %v", err)
		}
		if ret, err = a.ltDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lt reuse for string failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

/* Lte */

func TestDense_lteDD_Transitivity(t *testing.T) {
	fI := func(a, b, c *QCDenseI) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b, c *QCDenseI8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b, c *QCDenseI16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b, c *QCDenseI32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b, c *QCDenseI64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for int64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b, c *QCDenseU) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b, c *QCDenseU8) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint8 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint8 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint8 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b, c *QCDenseU16) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint16 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint16 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint16 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b, c *QCDenseU32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b, c *QCDenseU64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uint64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b, c *QCDenseUintptr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uintptr failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uintptr failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for uintptr failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b, c *QCDenseF32) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float32 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float32 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float32 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b, c *QCDenseF64) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float64 failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float64 failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for float64 failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b, c *QCDenseStr) bool {
		var axb, bxc, axc *Dense
		var err error
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte transitivity for string failed (axb) : %v ", err)
			return false
		}
		if bxc, err = b.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for string failed (bxc): %v", err)
			return false
		}
		if axc, err = a.lteDD(c.Dense); err != nil {
			t.Errorf("Test Lte transitivity for string failed (axc): %v", err)
			return false
		}

		ab := axb.bools()
		bc := bxc.bools()
		ac := axc.bools()

		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}
		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}

func Test_Dense_lteDD_funcOpts(t *testing.T) {
	fI := func(a, b *QCDenseI) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for int failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for int failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for int failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for int failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI, nil); err != nil {
		t.Error(err)
	}
	fI8 := func(a, b *QCDenseI8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int8, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for int8 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for int8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for int8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI8, nil); err != nil {
		t.Error(err)
	}
	fI16 := func(a, b *QCDenseI16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int16, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for int16 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for int16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for int16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI16, nil); err != nil {
		t.Error(err)
	}
	fI32 := func(a, b *QCDenseI32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int32, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for int32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for int32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for int32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI32, nil); err != nil {
		t.Error(err)
	}
	fI64 := func(a, b *QCDenseI64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Int64, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for int64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for int64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for int64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fI64, nil); err != nil {
		t.Error(err)
	}
	fU := func(a, b *QCDenseU) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for uint failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for uint failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for uint failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU, nil); err != nil {
		t.Error(err)
	}
	fU8 := func(a, b *QCDenseU8) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint8, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for uint8 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for uint8 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for uint8 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU8, nil); err != nil {
		t.Error(err)
	}
	fU16 := func(a, b *QCDenseU16) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint16, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for uint16 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for uint16 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for uint16 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU16, nil); err != nil {
		t.Error(err)
	}
	fU32 := func(a, b *QCDenseU32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint32, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for uint32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for uint32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for uint32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU32, nil); err != nil {
		t.Error(err)
	}
	fU64 := func(a, b *QCDenseU64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Uint64, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for uint64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for uint64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for uint64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fU64, nil); err != nil {
		t.Error(err)
	}
	fUintptr := func(a, b *QCDenseUintptr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for uintptr failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for uintptr failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fUintptr, nil); err != nil {
		t.Error(err)
	}
	fF32 := func(a, b *QCDenseF32) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float32, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for float32 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for float32 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for float32 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF32, nil); err != nil {
		t.Error(err)
	}
	fF64 := func(a, b *QCDenseF64) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		// reuse as same type
		reuse = recycledDense(Float64, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense, AsSameType()); err != nil {
			t.Errorf("Test Lte as same type reuse for float64 failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, AsSameType(), WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte as same type reuse for float64 failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error")
			return false
		}

		// unsafe
		if ret, err = a.lteDD(b.Dense, UseUnsafe()); err != nil {
			t.Errorf("Unsafe Lte for float64 failed %v", err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret to be equal to a")
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fF64, nil); err != nil {
		t.Error(err)
	}
	fStr := func(a, b *QCDenseStr) bool {
		var reuse, axb, ret *Dense
		var err error

		reuse = recycledDense(Bool, Shape{a.len()})
		if axb, err = a.lteDD(b.Dense); err != nil {
			t.Errorf("Test Lte reuse for string failed(axb): %v", err)
		}
		if ret, err = a.lteDD(b.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Test Lte reuse for string failed: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret to be equal reuse")
			return false
		}
		if !reflect.DeepEqual(axb.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(fStr, nil); err != nil {
		t.Error(err)
	}
}
