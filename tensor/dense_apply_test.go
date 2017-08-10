package tensor

import (
	"testing"
	"testing/quick"
	"unsafe"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestDense_Apply(t *testing.T) {
	idenB := func(a *QCDenseB) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Bool, a.len())
		correct.Memset(true)
		if ret, err = a.Apply(mutateB); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateB); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Bools()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Bools()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenB, nil); err != nil {
		t.Errorf("Applying mutation function to bool failed: %v", err)
	}

	// safety:
	unsafeB := func(a *QCDenseB) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Bool, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityB); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityB, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeB, nil); err != nil {
		t.Errorf("Unsafe identity function for bool failed %v", err)
	}

	idenI := func(a *QCDenseI) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Int, a.len())
		correct.Memset(int(1))
		if ret, err = a.Apply(mutateI); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateI); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Ints()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Ints()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Applying mutation function to int failed: %v", err)
	}

	// safety:
	unsafeI := func(a *QCDenseI) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityI); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityI, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeI, nil); err != nil {
		t.Errorf("Unsafe identity function for int failed %v", err)
	}

	incrI := func(a *QCDenseI) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityI, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Errorf("Applying identity function to int failed: %v", err)
	}

	idenI8 := func(a *QCDenseI8) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Int8, a.len())
		correct.Memset(int8(1))
		if ret, err = a.Apply(mutateI8); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateI8); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Int8s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Int8s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Applying mutation function to int8 failed: %v", err)
	}

	// safety:
	unsafeI8 := func(a *QCDenseI8) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityI8); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityI8, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeI8, nil); err != nil {
		t.Errorf("Unsafe identity function for int8 failed %v", err)
	}

	incrI8 := func(a *QCDenseI8) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityI8, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Errorf("Applying identity function to int8 failed: %v", err)
	}

	idenI16 := func(a *QCDenseI16) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Int16, a.len())
		correct.Memset(int16(1))
		if ret, err = a.Apply(mutateI16); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateI16); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Int16s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Int16s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Applying mutation function to int16 failed: %v", err)
	}

	// safety:
	unsafeI16 := func(a *QCDenseI16) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityI16); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityI16, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeI16, nil); err != nil {
		t.Errorf("Unsafe identity function for int16 failed %v", err)
	}

	incrI16 := func(a *QCDenseI16) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityI16, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Errorf("Applying identity function to int16 failed: %v", err)
	}

	idenI32 := func(a *QCDenseI32) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Int32, a.len())
		correct.Memset(int32(1))
		if ret, err = a.Apply(mutateI32); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateI32); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Int32s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Int32s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Applying mutation function to int32 failed: %v", err)
	}

	// safety:
	unsafeI32 := func(a *QCDenseI32) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityI32); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityI32, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeI32, nil); err != nil {
		t.Errorf("Unsafe identity function for int32 failed %v", err)
	}

	incrI32 := func(a *QCDenseI32) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityI32, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Errorf("Applying identity function to int32 failed: %v", err)
	}

	idenI64 := func(a *QCDenseI64) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Int64, a.len())
		correct.Memset(int64(1))
		if ret, err = a.Apply(mutateI64); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateI64); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Int64s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Int64s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Applying mutation function to int64 failed: %v", err)
	}

	// safety:
	unsafeI64 := func(a *QCDenseI64) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityI64); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityI64, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeI64, nil); err != nil {
		t.Errorf("Unsafe identity function for int64 failed %v", err)
	}

	incrI64 := func(a *QCDenseI64) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityI64, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Errorf("Applying identity function to int64 failed: %v", err)
	}

	idenU := func(a *QCDenseU) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uint, a.len())
		correct.Memset(uint(1))
		if ret, err = a.Apply(mutateU); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateU); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uints()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uints()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Applying mutation function to uint failed: %v", err)
	}

	// safety:
	unsafeU := func(a *QCDenseU) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityU); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityU, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeU, nil); err != nil {
		t.Errorf("Unsafe identity function for uint failed %v", err)
	}

	incrU := func(a *QCDenseU) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityU, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Errorf("Applying identity function to uint failed: %v", err)
	}

	idenU8 := func(a *QCDenseU8) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uint8, a.len())
		correct.Memset(uint8(1))
		if ret, err = a.Apply(mutateU8); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateU8); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uint8s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uint8s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Applying mutation function to uint8 failed: %v", err)
	}

	// safety:
	unsafeU8 := func(a *QCDenseU8) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityU8); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityU8, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeU8, nil); err != nil {
		t.Errorf("Unsafe identity function for uint8 failed %v", err)
	}

	incrU8 := func(a *QCDenseU8) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityU8, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Errorf("Applying identity function to uint8 failed: %v", err)
	}

	idenU16 := func(a *QCDenseU16) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uint16, a.len())
		correct.Memset(uint16(1))
		if ret, err = a.Apply(mutateU16); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateU16); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uint16s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uint16s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Applying mutation function to uint16 failed: %v", err)
	}

	// safety:
	unsafeU16 := func(a *QCDenseU16) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityU16); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityU16, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeU16, nil); err != nil {
		t.Errorf("Unsafe identity function for uint16 failed %v", err)
	}

	incrU16 := func(a *QCDenseU16) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityU16, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Errorf("Applying identity function to uint16 failed: %v", err)
	}

	idenU32 := func(a *QCDenseU32) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uint32, a.len())
		correct.Memset(uint32(1))
		if ret, err = a.Apply(mutateU32); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateU32); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uint32s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uint32s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Applying mutation function to uint32 failed: %v", err)
	}

	// safety:
	unsafeU32 := func(a *QCDenseU32) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityU32); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityU32, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeU32, nil); err != nil {
		t.Errorf("Unsafe identity function for uint32 failed %v", err)
	}

	incrU32 := func(a *QCDenseU32) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityU32, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Errorf("Applying identity function to uint32 failed: %v", err)
	}

	idenU64 := func(a *QCDenseU64) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uint64, a.len())
		correct.Memset(uint64(1))
		if ret, err = a.Apply(mutateU64); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateU64); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uint64s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uint64s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Applying mutation function to uint64 failed: %v", err)
	}

	// safety:
	unsafeU64 := func(a *QCDenseU64) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityU64); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityU64, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeU64, nil); err != nil {
		t.Errorf("Unsafe identity function for uint64 failed %v", err)
	}

	incrU64 := func(a *QCDenseU64) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityU64, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Errorf("Applying identity function to uint64 failed: %v", err)
	}

	idenUintptr := func(a *QCDenseUintptr) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Uintptr, a.len())
		correct.Memset(uintptr(0xdeadbeef))
		if ret, err = a.Apply(mutateUintptr); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateUintptr); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Uintptrs()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Uintptrs()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenUintptr, nil); err != nil {
		t.Errorf("Applying mutation function to uintptr failed: %v", err)
	}

	// safety:
	unsafeUintptr := func(a *QCDenseUintptr) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Uintptr, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityUintptr); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityUintptr, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeUintptr, nil); err != nil {
		t.Errorf("Unsafe identity function for uintptr failed %v", err)
	}

	idenF32 := func(a *QCDenseF32) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Float32, a.len())
		correct.Memset(float32(1))
		if ret, err = a.Apply(mutateF32); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateF32); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Float32s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Float32s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Applying mutation function to float32 failed: %v", err)
	}

	// safety:
	unsafeF32 := func(a *QCDenseF32) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityF32); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityF32, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeF32, nil); err != nil {
		t.Errorf("Unsafe identity function for float32 failed %v", err)
	}

	incrF32 := func(a *QCDenseF32) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityF32, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Errorf("Applying identity function to float32 failed: %v", err)
	}

	idenF64 := func(a *QCDenseF64) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Float64, a.len())
		correct.Memset(float64(1))
		if ret, err = a.Apply(mutateF64); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF32); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateF64); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Float64s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Float64s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF32); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Applying mutation function to float64 failed: %v", err)
	}

	// safety:
	unsafeF64 := func(a *QCDenseF64) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityF64); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityF64, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeF64, nil); err != nil {
		t.Errorf("Unsafe identity function for float64 failed %v", err)
	}

	incrF64 := func(a *QCDenseF64) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityF64, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Errorf("Applying identity function to float64 failed: %v", err)
	}

	idenC64 := func(a *QCDenseC64) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Complex64, a.len())
		correct.Memset(complex64(1))
		if ret, err = a.Apply(mutateC64); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateC64); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Complex64s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Complex64s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Applying mutation function to complex64 failed: %v", err)
	}

	// safety:
	unsafeC64 := func(a *QCDenseC64) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityC64); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityC64, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeC64, nil); err != nil {
		t.Errorf("Unsafe identity function for complex64 failed %v", err)
	}

	incrC64 := func(a *QCDenseC64) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityC64, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Errorf("Applying identity function to complex64 failed: %v", err)
	}

	idenC128 := func(a *QCDenseC128) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(Complex128, a.len())
		correct.Memset(complex128(1))
		if ret, err = a.Apply(mutateC128); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateC128); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Complex128s()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Complex128s()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Applying mutation function to complex128 failed: %v", err)
	}

	// safety:
	unsafeC128 := func(a *QCDenseC128) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityC128); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityC128, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeC128, nil); err != nil {
		t.Errorf("Unsafe identity function for complex128 failed %v", err)
	}

	incrC128 := func(a *QCDenseC128) bool {
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identityC128, WithIncr(a.Dense)); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Errorf("Applying identity function to complex128 failed: %v", err)
	}

	idenStr := func(a *QCDenseStr) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(String, a.len())
		correct.Memset("Hello World")
		if ret, err = a.Apply(mutateStr); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateStr); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.Strings()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.Strings()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenStr, nil); err != nil {
		t.Errorf("Applying mutation function to string failed: %v", err)
	}

	// safety:
	unsafeStr := func(a *QCDenseStr) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(String, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityStr); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityStr, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeStr, nil); err != nil {
		t.Errorf("Unsafe identity function for string failed %v", err)
	}

	idenUnsafePointer := func(a *QCDenseUnsafePointer) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense(UnsafePointer, a.len())
		correct.Memset(unsafe.Pointer(uintptr(0xdeadbeef)))
		if ret, err = a.Apply(mutateUnsafePointer); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply(identityF64); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutateUnsafePointer); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.UnsafePointers()[0:10]) {
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.UnsafePointers()[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply(identityF64); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(idenUnsafePointer, nil); err != nil {
		t.Errorf("Applying mutation function to unsafe.Pointer failed: %v", err)
	}

	// safety:
	unsafeUnsafePointer := func(a *QCDenseUnsafePointer) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense(UnsafePointer, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identityUnsafePointer); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identityUnsafePointer, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafeUnsafePointer, nil); err != nil {
		t.Errorf("Unsafe identity function for unsafe.Pointer failed %v", err)
	}

}
