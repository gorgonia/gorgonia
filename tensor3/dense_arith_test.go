package tensor

import (
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* Add */

func TestAddBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// commutativity
	commI := func(a, b *QCDenseI) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI, nil); err != nil {
		t.Errorf("Commutativity test for int failed %v", err)
	}
	// asociativity
	assocI := func(a, b, c *QCDenseI) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI, nil); err != nil {
		t.Errorf("Associativity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int8, a.len())
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// commutativity
	commI8 := func(a, b *QCDenseI8) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI8, nil); err != nil {
		t.Errorf("Commutativity test for int8 failed %v", err)
	}
	// asociativity
	assocI8 := func(a, b, c *QCDenseI8) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI8, nil); err != nil {
		t.Errorf("Associativity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int16, a.len())
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// commutativity
	commI16 := func(a, b *QCDenseI16) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI16, nil); err != nil {
		t.Errorf("Commutativity test for int16 failed %v", err)
	}
	// asociativity
	assocI16 := func(a, b, c *QCDenseI16) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI16, nil); err != nil {
		t.Errorf("Associativity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int32, a.len())
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// commutativity
	commI32 := func(a, b *QCDenseI32) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI32, nil); err != nil {
		t.Errorf("Commutativity test for int32 failed %v", err)
	}
	// asociativity
	assocI32 := func(a, b, c *QCDenseI32) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI32, nil); err != nil {
		t.Errorf("Associativity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int64, a.len())
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// commutativity
	commI64 := func(a, b *QCDenseI64) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI64, nil); err != nil {
		t.Errorf("Commutativity test for int64 failed %v", err)
	}
	// asociativity
	assocI64 := func(a, b, c *QCDenseI64) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI64, nil); err != nil {
		t.Errorf("Associativity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint, a.len())
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// commutativity
	commU := func(a, b *QCDenseU) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU, nil); err != nil {
		t.Errorf("Commutativity test for uint failed %v", err)
	}
	// asociativity
	assocU := func(a, b, c *QCDenseU) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU, nil); err != nil {
		t.Errorf("Associativity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint8, a.len())
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// commutativity
	commU8 := func(a, b *QCDenseU8) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU8, nil); err != nil {
		t.Errorf("Commutativity test for uint8 failed %v", err)
	}
	// asociativity
	assocU8 := func(a, b, c *QCDenseU8) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU8, nil); err != nil {
		t.Errorf("Associativity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint16, a.len())
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// commutativity
	commU16 := func(a, b *QCDenseU16) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU16, nil); err != nil {
		t.Errorf("Commutativity test for uint16 failed %v", err)
	}
	// asociativity
	assocU16 := func(a, b, c *QCDenseU16) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU16, nil); err != nil {
		t.Errorf("Associativity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint32, a.len())
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// commutativity
	commU32 := func(a, b *QCDenseU32) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU32, nil); err != nil {
		t.Errorf("Commutativity test for uint32 failed %v", err)
	}
	// asociativity
	assocU32 := func(a, b, c *QCDenseU32) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU32, nil); err != nil {
		t.Errorf("Associativity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint64, a.len())
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// commutativity
	commU64 := func(a, b *QCDenseU64) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU64, nil); err != nil {
		t.Errorf("Commutativity test for uint64 failed %v", err)
	}
	// asociativity
	assocU64 := func(a, b, c *QCDenseU64) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU64, nil); err != nil {
		t.Errorf("Associativity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float32, a.len())
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// commutativity
	commF32 := func(a, b *QCDenseF32) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF32, nil); err != nil {
		t.Errorf("Commutativity test for float32 failed %v", err)
	}
	// asociativity
	assocF32 := func(a, b, c *QCDenseF32) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocF32, nil); err != nil {
		t.Errorf("Associativity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// commutativity
	commF64 := func(a, b *QCDenseF64) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF64, nil); err != nil {
		t.Errorf("Commutativity test for float64 failed %v", err)
	}
	// asociativity
	assocF64 := func(a, b, c *QCDenseF64) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocF64, nil); err != nil {
		t.Errorf("Associativity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex64, a.len())
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// commutativity
	commC64 := func(a, b *QCDenseC64) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC64, nil); err != nil {
		t.Errorf("Commutativity test for complex64 failed %v", err)
	}
	// asociativity
	assocC64 := func(a, b, c *QCDenseC64) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocC64, nil); err != nil {
		t.Errorf("Associativity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex128, a.len())
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Add(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
	// commutativity
	commC128 := func(a, b *QCDenseC128) bool {
		ret1, _ := a.Add(b.Dense)
		ret2, _ := b.Add(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC128, nil); err != nil {
		t.Errorf("Commutativity test for complex128 failed %v", err)
	}
	// asociativity
	assocC128 := func(a, b, c *QCDenseC128) bool {
		ret1, _ := a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ := b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocC128, nil); err != nil {
		t.Errorf("Associativity test for complex128 failed %v", err)
	}
}
func TestAddFuncOpts(t *testing.T) {
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		if _, err := a.Add(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for Add failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		if ret, err = a.Add(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Add")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for Add failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.Add(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.Add(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Add using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.Add(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Add using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for Add failed : %v ", err)
	}

	// unsafe
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.Add(identity, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for Add failed : %v ", err)
	}
}

/* Sub */

func TestSubBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int8, a.len())
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int16, a.len())
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int32, a.len())
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int64, a.len())
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint, a.len())
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint8, a.len())
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint16, a.len())
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint32, a.len())
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint64, a.len())
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float32, a.len())
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex64, a.len())
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex128, a.len())
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Sub(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}
func TestSubFuncOpts(t *testing.T) {
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		if _, err := a.Sub(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for Sub failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		if ret, err = a.Sub(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Sub")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for Sub failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.Sub(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.Sub(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Sub using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.Sub(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Sub using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for Sub failed : %v ", err)
	}

	// unsafe
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.Sub(identity, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for Sub failed : %v ", err)
	}
}

/* Mul */

func TestMulBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int, a.len())
		identity.Memset(int(1))
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// commutativity
	commI := func(a, b *QCDenseI) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI, nil); err != nil {
		t.Errorf("Commutativity test for int failed %v", err)
	}
	// asociativity
	assocI := func(a, b, c *QCDenseI) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI, nil); err != nil {
		t.Errorf("Associativity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int8, a.len())
		identity.Memset(int8(1))
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// commutativity
	commI8 := func(a, b *QCDenseI8) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI8, nil); err != nil {
		t.Errorf("Commutativity test for int8 failed %v", err)
	}
	// asociativity
	assocI8 := func(a, b, c *QCDenseI8) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI8, nil); err != nil {
		t.Errorf("Associativity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int16, a.len())
		identity.Memset(int16(1))
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// commutativity
	commI16 := func(a, b *QCDenseI16) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI16, nil); err != nil {
		t.Errorf("Commutativity test for int16 failed %v", err)
	}
	// asociativity
	assocI16 := func(a, b, c *QCDenseI16) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI16, nil); err != nil {
		t.Errorf("Associativity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int32, a.len())
		identity.Memset(int32(1))
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// commutativity
	commI32 := func(a, b *QCDenseI32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI32, nil); err != nil {
		t.Errorf("Commutativity test for int32 failed %v", err)
	}
	// asociativity
	assocI32 := func(a, b, c *QCDenseI32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI32, nil); err != nil {
		t.Errorf("Associativity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int64, a.len())
		identity.Memset(int64(1))
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// commutativity
	commI64 := func(a, b *QCDenseI64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI64, nil); err != nil {
		t.Errorf("Commutativity test for int64 failed %v", err)
	}
	// asociativity
	assocI64 := func(a, b, c *QCDenseI64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocI64, nil); err != nil {
		t.Errorf("Associativity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint, a.len())
		identity.Memset(uint(1))
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// commutativity
	commU := func(a, b *QCDenseU) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU, nil); err != nil {
		t.Errorf("Commutativity test for uint failed %v", err)
	}
	// asociativity
	assocU := func(a, b, c *QCDenseU) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU, nil); err != nil {
		t.Errorf("Associativity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint8, a.len())
		identity.Memset(uint8(1))
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// commutativity
	commU8 := func(a, b *QCDenseU8) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU8, nil); err != nil {
		t.Errorf("Commutativity test for uint8 failed %v", err)
	}
	// asociativity
	assocU8 := func(a, b, c *QCDenseU8) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU8, nil); err != nil {
		t.Errorf("Associativity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint16, a.len())
		identity.Memset(uint16(1))
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// commutativity
	commU16 := func(a, b *QCDenseU16) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU16, nil); err != nil {
		t.Errorf("Commutativity test for uint16 failed %v", err)
	}
	// asociativity
	assocU16 := func(a, b, c *QCDenseU16) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU16, nil); err != nil {
		t.Errorf("Associativity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint32, a.len())
		identity.Memset(uint32(1))
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// commutativity
	commU32 := func(a, b *QCDenseU32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU32, nil); err != nil {
		t.Errorf("Commutativity test for uint32 failed %v", err)
	}
	// asociativity
	assocU32 := func(a, b, c *QCDenseU32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU32, nil); err != nil {
		t.Errorf("Associativity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint64, a.len())
		identity.Memset(uint64(1))
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// commutativity
	commU64 := func(a, b *QCDenseU64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU64, nil); err != nil {
		t.Errorf("Commutativity test for uint64 failed %v", err)
	}
	// asociativity
	assocU64 := func(a, b, c *QCDenseU64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocU64, nil); err != nil {
		t.Errorf("Associativity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float32, a.len())
		identity.Memset(float32(1))
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// commutativity
	commF32 := func(a, b *QCDenseF32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF32, nil); err != nil {
		t.Errorf("Commutativity test for float32 failed %v", err)
	}
	// asociativity
	assocF32 := func(a, b, c *QCDenseF32) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocF32, nil); err != nil {
		t.Errorf("Associativity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// commutativity
	commF64 := func(a, b *QCDenseF64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF64, nil); err != nil {
		t.Errorf("Commutativity test for float64 failed %v", err)
	}
	// asociativity
	assocF64 := func(a, b, c *QCDenseF64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocF64, nil); err != nil {
		t.Errorf("Associativity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex64, a.len())
		identity.Memset(complex64(1))
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// commutativity
	commC64 := func(a, b *QCDenseC64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC64, nil); err != nil {
		t.Errorf("Commutativity test for complex64 failed %v", err)
	}
	// asociativity
	assocC64 := func(a, b, c *QCDenseC64) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocC64, nil); err != nil {
		t.Errorf("Associativity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex128, a.len())
		identity.Memset(complex128(1))
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Mul(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
	// commutativity
	commC128 := func(a, b *QCDenseC128) bool {
		ret1, _ := a.Mul(b.Dense)
		ret2, _ := b.Mul(a.Dense)
		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC128, nil); err != nil {
		t.Errorf("Commutativity test for complex128 failed %v", err)
	}
	// asociativity
	assocC128 := func(a, b, c *QCDenseC128) bool {
		ret1, _ := a.Mul(b.Dense)
		ret1, _ = ret1.Mul(c.Dense)

		ret2, _ := b.Mul(c.Dense)
		ret2, _ = a.Mul(ret2)

		if !allClose(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(assocC128, nil); err != nil {
		t.Errorf("Associativity test for complex128 failed %v", err)
	}
}
func TestMulFuncOpts(t *testing.T) {
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		identity.Memset(1)
		if _, err := a.Mul(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for Mul failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(1)
		if ret, err = a.Mul(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Mul")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for Mul failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.Mul(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.Mul(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Mul using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.Mul(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Mul using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for Mul failed : %v ", err)
	}

	// unsafe
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.Mul(identity, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for Mul failed : %v ", err)
	}
}

/* Div */

func TestDivBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int, a.len())
		identity.Memset(int(1))
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int8, a.len())
		identity.Memset(int8(1))
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int16, a.len())
		identity.Memset(int16(1))
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int32, a.len())
		identity.Memset(int32(1))
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int64, a.len())
		identity.Memset(int64(1))
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint, a.len())
		identity.Memset(uint(1))
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint8, a.len())
		identity.Memset(uint8(1))
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint16, a.len())
		identity.Memset(uint16(1))
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint32, a.len())
		identity.Memset(uint32(1))
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint64, a.len())
		identity.Memset(uint64(1))
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float32, a.len())
		identity.Memset(float32(1))
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex64, a.len())
		identity.Memset(complex64(1))
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex128, a.len())
		identity.Memset(complex128(1))
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Div(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}
func TestDivFuncOpts(t *testing.T) {
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		identity.Memset(1)
		if _, err := a.Div(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for Div failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(1)
		if ret, err = a.Div(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Div")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for Div failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.Div(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.Div(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Div using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.Div(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Div using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for Div failed : %v ", err)
	}

	// unsafe
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.Div(identity, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for Div failed : %v ", err)
	}
}

/* Pow */

func TestPowBasicProperties(t *testing.T) {
	pow0I := func(a *QCDenseI) bool {
		var ret, correct, zero *Dense
		zero = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		correct.Memset(int(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I8 := func(a *QCDenseI8) bool {
		var ret, correct, zero *Dense
		zero = newDense(Int8, a.len())
		correct = newDense(Int8, a.len())
		correct.Memset(int8(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I16 := func(a *QCDenseI16) bool {
		var ret, correct, zero *Dense
		zero = newDense(Int16, a.len())
		correct = newDense(Int16, a.len())
		correct.Memset(int16(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I32 := func(a *QCDenseI32) bool {
		var ret, correct, zero *Dense
		zero = newDense(Int32, a.len())
		correct = newDense(Int32, a.len())
		correct.Memset(int32(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I64 := func(a *QCDenseI64) bool {
		var ret, correct, zero *Dense
		zero = newDense(Int64, a.len())
		correct = newDense(Int64, a.len())
		correct.Memset(int64(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U := func(a *QCDenseU) bool {
		var ret, correct, zero *Dense
		zero = newDense(Uint, a.len())
		correct = newDense(Uint, a.len())
		correct.Memset(uint(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U8 := func(a *QCDenseU8) bool {
		var ret, correct, zero *Dense
		zero = newDense(Uint8, a.len())
		correct = newDense(Uint8, a.len())
		correct.Memset(uint8(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U16 := func(a *QCDenseU16) bool {
		var ret, correct, zero *Dense
		zero = newDense(Uint16, a.len())
		correct = newDense(Uint16, a.len())
		correct.Memset(uint16(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U32 := func(a *QCDenseU32) bool {
		var ret, correct, zero *Dense
		zero = newDense(Uint32, a.len())
		correct = newDense(Uint32, a.len())
		correct.Memset(uint32(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U64 := func(a *QCDenseU64) bool {
		var ret, correct, zero *Dense
		zero = newDense(Uint64, a.len())
		correct = newDense(Uint64, a.len())
		correct.Memset(uint64(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0F32 := func(a *QCDenseF32) bool {
		var ret, correct, zero *Dense
		zero = newDense(Float32, a.len())
		correct = newDense(Float32, a.len())
		correct.Memset(float32(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0F32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0F64 := func(a *QCDenseF64) bool {
		var ret, correct, zero *Dense
		zero = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		correct.Memset(float64(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0F64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0C64 := func(a *QCDenseC64) bool {
		var ret, correct, zero *Dense
		zero = newDense(Complex64, a.len())
		correct = newDense(Complex64, a.len())
		correct.Memset(complex64(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0C64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0C128 := func(a *QCDenseC128) bool {
		var ret, correct, zero *Dense
		zero = newDense(Complex128, a.len())
		correct = newDense(Complex128, a.len())
		correct.Memset(complex128(1))
		ret, _ = a.Pow(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0C128, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
}
func TestPowFuncOpts(t *testing.T) {
	var f func(*QCDenseF64) bool

	f = func(a *QCDenseF64) bool {
		identity := newDense(Float64, a.len()+1)
		identity.Memset(1)
		if _, err := a.Pow(identity); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("len test for Pow failed : %v ", err)
	}

	// safe
	f = func(a *QCDenseF64) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(1)
		if ret, err = a.Pow(identity); err != nil {
			t.Error(err)
			return false
		}
		if ret == identity || ret == a.Dense {
			t.Errorf("Failed safe test for Pow")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("safe test for Pow failed : %v ", err)
	}

	// reuse
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		reuse = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		if ret, err = a.Pow(identity, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Errorf("Expected ret %p == reuse %p", ret, reuse)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Expected Reuse: %v\nGot reuse : %v", correct.Data(), ret.Data())
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = a.Pow(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Pow using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Float64, a.len()+1)
		if _, err = a.Pow(identity, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when doing Pow using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("reuse test for Pow failed : %v ", err)
	}

	// unsafe
	f = func(a *QCDenseF64) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Float64, a.len())
		identity.Memset(float64(1))
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		if ret, err = a.Pow(identity, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Errorf("unsafe test for Pow failed : %v ", err)
	}
}

/* Trans */

func TestTransBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct *Dense
		var identity int
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct *Dense
		var identity int8
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct *Dense
		var identity int16
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct *Dense
		var identity int32
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct *Dense
		var identity int64
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct *Dense
		var identity uint
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct *Dense
		var identity uint8
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct *Dense
		var identity uint16
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct *Dense
		var identity uint32
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct *Dense
		var identity uint64
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct *Dense
		var identity float32
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct *Dense
		var identity float64
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct *Dense
		var identity complex64
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct *Dense
		var identity complex128
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Trans(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}

/* TransInv */

func TestTransInvBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct *Dense
		var identity int
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct *Dense
		var identity int8
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct *Dense
		var identity int16
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct *Dense
		var identity int32
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct *Dense
		var identity int64
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct *Dense
		var identity uint
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct *Dense
		var identity uint8
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct *Dense
		var identity uint16
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct *Dense
		var identity uint32
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct *Dense
		var identity uint64
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct *Dense
		var identity float32
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct *Dense
		var identity float64
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct *Dense
		var identity complex64
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct *Dense
		var identity complex128
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.TransInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}

/* TransInvR */

func TestTransInvRBasicProperties(t *testing.T) {
}

/* Scale */

func TestScaleBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct *Dense
		var identity int
		identity = 1
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct *Dense
		var identity int8
		identity = 1
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct *Dense
		var identity int16
		identity = 1
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct *Dense
		var identity int32
		identity = 1
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct *Dense
		var identity int64
		identity = 1
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct *Dense
		var identity uint
		identity = 1
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct *Dense
		var identity uint8
		identity = 1
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct *Dense
		var identity uint16
		identity = 1
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct *Dense
		var identity uint32
		identity = 1
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct *Dense
		var identity uint64
		identity = 1
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct *Dense
		var identity float32
		identity = 1
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct *Dense
		var identity float64
		identity = 1
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct *Dense
		var identity complex64
		identity = 1
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct *Dense
		var identity complex128
		identity = 1
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.Scale(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}

/* ScaleInv */

func TestScaleInvBasicProperties(t *testing.T) {
	// identity
	idenI := func(a *QCDenseI) bool {
		var ret, correct *Dense
		var identity int
		identity = 1
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// identity
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct *Dense
		var identity int8
		identity = 1
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// identity
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct *Dense
		var identity int16
		identity = 1
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// identity
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct *Dense
		var identity int32
		identity = 1
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// identity
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct *Dense
		var identity int64
		identity = 1
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// identity
	idenU := func(a *QCDenseU) bool {
		var ret, correct *Dense
		var identity uint
		identity = 1
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// identity
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct *Dense
		var identity uint8
		identity = 1
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// identity
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct *Dense
		var identity uint16
		identity = 1
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// identity
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct *Dense
		var identity uint32
		identity = 1
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// identity
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct *Dense
		var identity uint64
		identity = 1
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// identity
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct *Dense
		var identity float32
		identity = 1
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// identity
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct *Dense
		var identity float64
		identity = 1
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// identity
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct *Dense
		var identity complex64
		identity = 1
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// identity
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct *Dense
		var identity complex128
		identity = 1
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = a.ScaleInv(identity)

		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
}

/* ScaleInvR */

func TestScaleInvRBasicProperties(t *testing.T) {
}

/* PowOf */

func TestPowOfBasicProperties(t *testing.T) {
	pow0I := func(a *QCDenseI) bool {
		var ret, correct *Dense
		var zero int
		correct = newDense(Int, a.len())
		correct.Memset(int(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I8 := func(a *QCDenseI8) bool {
		var ret, correct *Dense
		var zero int8
		correct = newDense(Int8, a.len())
		correct.Memset(int8(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I16 := func(a *QCDenseI16) bool {
		var ret, correct *Dense
		var zero int16
		correct = newDense(Int16, a.len())
		correct.Memset(int16(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I32 := func(a *QCDenseI32) bool {
		var ret, correct *Dense
		var zero int32
		correct = newDense(Int32, a.len())
		correct.Memset(int32(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0I64 := func(a *QCDenseI64) bool {
		var ret, correct *Dense
		var zero int64
		correct = newDense(Int64, a.len())
		correct.Memset(int64(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0I64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U := func(a *QCDenseU) bool {
		var ret, correct *Dense
		var zero uint
		correct = newDense(Uint, a.len())
		correct.Memset(uint(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U8 := func(a *QCDenseU8) bool {
		var ret, correct *Dense
		var zero uint8
		correct = newDense(Uint8, a.len())
		correct.Memset(uint8(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U16 := func(a *QCDenseU16) bool {
		var ret, correct *Dense
		var zero uint16
		correct = newDense(Uint16, a.len())
		correct.Memset(uint16(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U32 := func(a *QCDenseU32) bool {
		var ret, correct *Dense
		var zero uint32
		correct = newDense(Uint32, a.len())
		correct.Memset(uint32(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0U64 := func(a *QCDenseU64) bool {
		var ret, correct *Dense
		var zero uint64
		correct = newDense(Uint64, a.len())
		correct.Memset(uint64(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0U64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0F32 := func(a *QCDenseF32) bool {
		var ret, correct *Dense
		var zero float32
		correct = newDense(Float32, a.len())
		correct.Memset(float32(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0F32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0F64 := func(a *QCDenseF64) bool {
		var ret, correct *Dense
		var zero float64
		correct = newDense(Float64, a.len())
		correct.Memset(float64(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0F64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0C64 := func(a *QCDenseC64) bool {
		var ret, correct *Dense
		var zero complex64
		correct = newDense(Complex64, a.len())
		correct.Memset(complex64(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0C64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	pow0C128 := func(a *QCDenseC128) bool {
		var ret, correct *Dense
		var zero complex128
		correct = newDense(Complex128, a.len())
		correct.Memset(complex128(1))
		ret, _ = a.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0C128, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
}

/* PowOfR */

func TestPowOfRBasicProperties(t *testing.T) {
}
