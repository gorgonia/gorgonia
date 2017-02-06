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

	idenSlicedI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int, 5)
		correct = newDense(Int, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI, nil); err != nil {
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
	// incr
	incrI := func(a, b, incr *QCDenseI) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenSlicedI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int8, 5)
		correct = newDense(Int8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI8, nil); err != nil {
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
	// incr
	incrI8 := func(a, b, incr *QCDenseI8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenSlicedI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int16, 5)
		correct = newDense(Int16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI16, nil); err != nil {
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
	// incr
	incrI16 := func(a, b, incr *QCDenseI16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenSlicedI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int32, 5)
		correct = newDense(Int32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI32, nil); err != nil {
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
	// incr
	incrI32 := func(a, b, incr *QCDenseI32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenSlicedI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int64, 5)
		correct = newDense(Int64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI64, nil); err != nil {
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
	// incr
	incrI64 := func(a, b, incr *QCDenseI64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenSlicedU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint, 5)
		correct = newDense(Uint, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU, nil); err != nil {
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
	// incr
	incrU := func(a, b, incr *QCDenseU) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenSlicedU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint8, 5)
		correct = newDense(Uint8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU8, nil); err != nil {
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
	// incr
	incrU8 := func(a, b, incr *QCDenseU8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenSlicedU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint16, 5)
		correct = newDense(Uint16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU16, nil); err != nil {
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
	// incr
	incrU16 := func(a, b, incr *QCDenseU16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenSlicedU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint32, 5)
		correct = newDense(Uint32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU32, nil); err != nil {
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
	// incr
	incrU32 := func(a, b, incr *QCDenseU32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenSlicedU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint64, 5)
		correct = newDense(Uint64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU64, nil); err != nil {
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
	// incr
	incrU64 := func(a, b, incr *QCDenseU64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenSlicedF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float32, 5)
		correct = newDense(Float32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF32, nil); err != nil {
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
	// incr
	incrF32 := func(a, b, incr *QCDenseF32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenSlicedF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float64, 5)
		correct = newDense(Float64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF64, nil); err != nil {
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
	// incr
	incrF64 := func(a, b, incr *QCDenseF64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenSlicedC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex64, 5)
		correct = newDense(Complex64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC64, nil); err != nil {
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
	// incr
	incrC64 := func(a, b, incr *QCDenseC64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenSlicedC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex128, 5)
		correct = newDense(Complex128, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex128, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Add(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC128, nil); err != nil {
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
	// incr
	incrC128 := func(a, b, incr *QCDenseC128) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Add(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Add(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Add(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Add(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenSlicedI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int, 5)
		correct = newDense(Int, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// incr
	incrI := func(a, b, incr *QCDenseI) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenSlicedI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int8, 5)
		correct = newDense(Int8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// incr
	incrI8 := func(a, b, incr *QCDenseI8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenSlicedI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int16, 5)
		correct = newDense(Int16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// incr
	incrI16 := func(a, b, incr *QCDenseI16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenSlicedI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int32, 5)
		correct = newDense(Int32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// incr
	incrI32 := func(a, b, incr *QCDenseI32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenSlicedI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int64, 5)
		correct = newDense(Int64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// incr
	incrI64 := func(a, b, incr *QCDenseI64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenSlicedU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint, 5)
		correct = newDense(Uint, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// incr
	incrU := func(a, b, incr *QCDenseU) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenSlicedU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint8, 5)
		correct = newDense(Uint8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// incr
	incrU8 := func(a, b, incr *QCDenseU8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenSlicedU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint16, 5)
		correct = newDense(Uint16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// incr
	incrU16 := func(a, b, incr *QCDenseU16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenSlicedU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint32, 5)
		correct = newDense(Uint32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// incr
	incrU32 := func(a, b, incr *QCDenseU32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenSlicedU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint64, 5)
		correct = newDense(Uint64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// incr
	incrU64 := func(a, b, incr *QCDenseU64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenSlicedF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float32, 5)
		correct = newDense(Float32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// incr
	incrF32 := func(a, b, incr *QCDenseF32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenSlicedF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float64, 5)
		correct = newDense(Float64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// incr
	incrF64 := func(a, b, incr *QCDenseF64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenSlicedC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex64, 5)
		correct = newDense(Complex64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// incr
	incrC64 := func(a, b, incr *QCDenseC64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenSlicedC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex128, 5)
		correct = newDense(Complex128, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex128, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		ret, _ = a2.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Sub(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
	// incr
	incrC128 := func(a, b, incr *QCDenseC128) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Sub(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Sub(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Sub(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Sub(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenSlicedI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int, 5)
		identity.Memset(int(1))
		correct = newDense(Int, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI, nil); err != nil {
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
	// incr
	incrI := func(a, b, incr *QCDenseI) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenSlicedI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int8, 5)
		identity.Memset(int8(1))
		correct = newDense(Int8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int8(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI8, nil); err != nil {
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
	// incr
	incrI8 := func(a, b, incr *QCDenseI8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenSlicedI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int16, 5)
		identity.Memset(int16(1))
		correct = newDense(Int16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int16(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI16, nil); err != nil {
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
	// incr
	incrI16 := func(a, b, incr *QCDenseI16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenSlicedI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int32, 5)
		identity.Memset(int32(1))
		correct = newDense(Int32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int32(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI32, nil); err != nil {
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
	// incr
	incrI32 := func(a, b, incr *QCDenseI32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenSlicedI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int64, 5)
		identity.Memset(int64(1))
		correct = newDense(Int64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int64(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI64, nil); err != nil {
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
	// incr
	incrI64 := func(a, b, incr *QCDenseI64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenSlicedU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint, 5)
		identity.Memset(uint(1))
		correct = newDense(Uint, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU, nil); err != nil {
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
	// incr
	incrU := func(a, b, incr *QCDenseU) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenSlicedU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint8, 5)
		identity.Memset(uint8(1))
		correct = newDense(Uint8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint8(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU8, nil); err != nil {
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
	// incr
	incrU8 := func(a, b, incr *QCDenseU8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenSlicedU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint16, 5)
		identity.Memset(uint16(1))
		correct = newDense(Uint16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint16(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU16, nil); err != nil {
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
	// incr
	incrU16 := func(a, b, incr *QCDenseU16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenSlicedU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint32, 5)
		identity.Memset(uint32(1))
		correct = newDense(Uint32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint32(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU32, nil); err != nil {
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
	// incr
	incrU32 := func(a, b, incr *QCDenseU32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenSlicedU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint64, 5)
		identity.Memset(uint64(1))
		correct = newDense(Uint64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint64(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU64, nil); err != nil {
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
	// incr
	incrU64 := func(a, b, incr *QCDenseU64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenSlicedF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float32, 5)
		identity.Memset(float32(1))
		correct = newDense(Float32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(float32(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF32, nil); err != nil {
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
	// incr
	incrF32 := func(a, b, incr *QCDenseF32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenSlicedF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float64, 5)
		identity.Memset(float64(1))
		correct = newDense(Float64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(float64(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF64, nil); err != nil {
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
	// incr
	incrF64 := func(a, b, incr *QCDenseF64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenSlicedC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex64, 5)
		identity.Memset(complex64(1))
		correct = newDense(Complex64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(complex64(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC64, nil); err != nil {
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
	// incr
	incrC64 := func(a, b, incr *QCDenseC64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenSlicedC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex128, 5)
		identity.Memset(complex128(1))
		correct = newDense(Complex128, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex128, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(complex128(1))
		ret, _ = a2.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Mul(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC128, nil); err != nil {
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
	// incr
	incrC128 := func(a, b, incr *QCDenseC128) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Mul(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Mul(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Mul(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Mul(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenSlicedI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int, 5)
		identity.Memset(int(1))
		correct = newDense(Int, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI, nil); err != nil {
		t.Errorf("Identity test for int failed %v", err)
	}
	// incr
	incrI := func(a, b, incr *QCDenseI) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(int(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenSlicedI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int8, 5)
		identity.Memset(int8(1))
		correct = newDense(Int8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int8(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI8, nil); err != nil {
		t.Errorf("Identity test for int8 failed %v", err)
	}
	// incr
	incrI8 := func(a, b, incr *QCDenseI8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(int8(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenSlicedI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int16, 5)
		identity.Memset(int16(1))
		correct = newDense(Int16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int16(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI16, nil); err != nil {
		t.Errorf("Identity test for int16 failed %v", err)
	}
	// incr
	incrI16 := func(a, b, incr *QCDenseI16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(int16(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenSlicedI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int32, 5)
		identity.Memset(int32(1))
		correct = newDense(Int32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int32(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI32, nil); err != nil {
		t.Errorf("Identity test for int32 failed %v", err)
	}
	// incr
	incrI32 := func(a, b, incr *QCDenseI32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(int32(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenSlicedI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Int64, 5)
		identity.Memset(int64(1))
		correct = newDense(Int64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Int64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(int64(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedI64, nil); err != nil {
		t.Errorf("Identity test for int64 failed %v", err)
	}
	// incr
	incrI64 := func(a, b, incr *QCDenseI64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(int64(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenSlicedU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint, 5)
		identity.Memset(uint(1))
		correct = newDense(Uint, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU, nil); err != nil {
		t.Errorf("Identity test for uint failed %v", err)
	}
	// incr
	incrU := func(a, b, incr *QCDenseU) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(uint(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenSlicedU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint8, 5)
		identity.Memset(uint8(1))
		correct = newDense(Uint8, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint8, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint8(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU8, nil); err != nil {
		t.Errorf("Identity test for uint8 failed %v", err)
	}
	// incr
	incrU8 := func(a, b, incr *QCDenseU8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(uint8(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenSlicedU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint16, 5)
		identity.Memset(uint16(1))
		correct = newDense(Uint16, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint16, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint16(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU16, nil); err != nil {
		t.Errorf("Identity test for uint16 failed %v", err)
	}
	// incr
	incrU16 := func(a, b, incr *QCDenseU16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(uint16(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenSlicedU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint32, 5)
		identity.Memset(uint32(1))
		correct = newDense(Uint32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint32(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU32, nil); err != nil {
		t.Errorf("Identity test for uint32 failed %v", err)
	}
	// incr
	incrU32 := func(a, b, incr *QCDenseU32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(uint32(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenSlicedU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Uint64, 5)
		identity.Memset(uint64(1))
		correct = newDense(Uint64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Uint64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(uint64(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedU64, nil); err != nil {
		t.Errorf("Identity test for uint64 failed %v", err)
	}
	// incr
	incrU64 := func(a, b, incr *QCDenseU64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(uint64(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenSlicedF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float32, 5)
		identity.Memset(float32(1))
		correct = newDense(Float32, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float32, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(float32(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF32, nil); err != nil {
		t.Errorf("Identity test for float32 failed %v", err)
	}
	// incr
	incrF32 := func(a, b, incr *QCDenseF32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(float32(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenSlicedF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Float64, 5)
		identity.Memset(float64(1))
		correct = newDense(Float64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Float64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(float64(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedF64, nil); err != nil {
		t.Errorf("Identity test for float64 failed %v", err)
	}
	// incr
	incrF64 := func(a, b, incr *QCDenseF64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(float64(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenSlicedC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex64, 5)
		identity.Memset(complex64(1))
		correct = newDense(Complex64, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex64, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(complex64(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC64, nil); err != nil {
		t.Errorf("Identity test for complex64 failed %v", err)
	}
	// incr
	incrC64 := func(a, b, incr *QCDenseC64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(complex64(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenSlicedC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity = newDense(Complex128, 5)
		identity.Memset(complex128(1))
		correct = newDense(Complex128, 5)
		copyDense(correct, a.Dense)

		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// other requires iterator
		a2 := a1.Materialize().(*Dense)
		identity = newDense(Complex128, a.len())
		identity, _ = sliceDense(identity, makeRS(0, 5))
		identity.Memset(complex128(1))
		ret, _ = a2.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		ret, _ = a1.Div(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenSlicedC128, nil); err != nil {
		t.Errorf("Identity test for complex128 failed %v", err)
	}
	// incr
	incrC128 := func(a, b, incr *QCDenseC128) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		b.Dense.Memset(complex128(1))
		ret, _ = a.Div(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Div(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Div(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Div(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	pow0IterI := func(a *QCDenseI) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Int, 5)
		correct = newDense(Int, 5)
		correct.Memset(int(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Int, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterI, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrI := func(a, b, incr *QCDenseI) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	pow0IterI8 := func(a *QCDenseI8) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Int8, 5)
		correct = newDense(Int8, 5)
		correct.Memset(int8(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Int8, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterI8, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrI8 := func(a, b, incr *QCDenseI8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	pow0IterI16 := func(a *QCDenseI16) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Int16, 5)
		correct = newDense(Int16, 5)
		correct.Memset(int16(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Int16, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterI16, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrI16 := func(a, b, incr *QCDenseI16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	pow0IterI32 := func(a *QCDenseI32) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Int32, 5)
		correct = newDense(Int32, 5)
		correct.Memset(int32(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Int32, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterI32, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrI32 := func(a, b, incr *QCDenseI32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	pow0IterI64 := func(a *QCDenseI64) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Int64, 5)
		correct = newDense(Int64, 5)
		correct.Memset(int64(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Int64, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterI64, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrI64 := func(a, b, incr *QCDenseI64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	pow0IterU := func(a *QCDenseU) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Uint, 5)
		correct = newDense(Uint, 5)
		correct.Memset(uint(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Uint, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterU, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrU := func(a, b, incr *QCDenseU) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	pow0IterU8 := func(a *QCDenseU8) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Uint8, 5)
		correct = newDense(Uint8, 5)
		correct.Memset(uint8(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Uint8, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterU8, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrU8 := func(a, b, incr *QCDenseU8) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	pow0IterU16 := func(a *QCDenseU16) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Uint16, 5)
		correct = newDense(Uint16, 5)
		correct.Memset(uint16(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Uint16, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterU16, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrU16 := func(a, b, incr *QCDenseU16) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	pow0IterU32 := func(a *QCDenseU32) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Uint32, 5)
		correct = newDense(Uint32, 5)
		correct.Memset(uint32(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Uint32, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterU32, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrU32 := func(a, b, incr *QCDenseU32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	pow0IterU64 := func(a *QCDenseU64) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Uint64, 5)
		correct = newDense(Uint64, 5)
		correct.Memset(uint64(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Uint64, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterU64, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrU64 := func(a, b, incr *QCDenseU64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	pow0IterF32 := func(a *QCDenseF32) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Float32, 5)
		correct = newDense(Float32, 5)
		correct.Memset(float32(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Float32, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterF32, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrF32 := func(a, b, incr *QCDenseF32) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	pow0IterF64 := func(a *QCDenseF64) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Float64, 5)
		correct = newDense(Float64, 5)
		correct.Memset(float64(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Float64, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterF64, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrF64 := func(a, b, incr *QCDenseF64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	pow0IterC64 := func(a *QCDenseC64) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Complex64, 5)
		correct = newDense(Complex64, 5)
		correct.Memset(complex64(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Complex64, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterC64, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrC64 := func(a, b, incr *QCDenseC64) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	pow0IterC128 := func(a *QCDenseC128) bool {
		var ret, correct, zero *Dense

		// t requires iterator
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		zero = newDense(Complex128, 5)
		correct = newDense(Complex128, 5)
		correct.Memset(complex128(1))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// zero requires iterator
		a2 := a1.Materialize().(*Dense)
		zero = newDense(Complex128, a.len())
		zero, _ = sliceDense(zero, makeRS(0, 5))
		ret, _ = a2.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// both requires iterator
		a1, _ = sliceDense(a.Dense, makeRS(6, 11))
		ret, _ = a1.Pow(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(pow0IterC128, nil); err != nil {
		t.Errorf("Pow 0 with iterator failed")
	}

	// incr
	incrC128 := func(a, b, incr *QCDenseC128) bool {
		var correct, clonedIncr, ret, check *Dense

		// build correct
		ret, _ = a.Pow(b.Dense)
		correct, _ = incr.Add(ret)

		clonedIncr = incr.Clone().(*Dense)
		check, _ = a.Pow(b.Dense, WithIncr(clonedIncr))
		if check != clonedIncr {
			t.Error("Expected clonedIncr == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Failed close")
			return false
		}

		// incr iter
		var oncr, a1, a2, b1, b2 *Dense
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		a2 = a1.Materialize().(*Dense)
		b1, _ = sliceDense(b.Dense, makeRS(0, 5))
		b2 = b1.Materialize().(*Dense)
		// build correct for incr
		correct, _ = sliceDense(correct, makeRS(0, 5))

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check both don't require iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		check, _ = a2.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// incr noiter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		correct = correct.Materialize().(*Dense)

		// check: a requires iter
		check, _ = a1.Pow(b2, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when a requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a2.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		// check: both a and b requires iter
		clonedIncr = incr.Dense.Clone().(*Dense)
		oncr, _ = sliceDense(clonedIncr, makeRS(0, 5))
		oncr = oncr.Materialize().(*Dense)
		check, _ = a1.Pow(b1, WithIncr(oncr))
		if check != oncr {
			t.Errorf("expected check == oncr when b requires iter")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenIterI := func(a *QCDenseI) bool {
		var a1, ret, correct *Dense
		var identity int
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI, nil); err != nil {
		t.Errorf("Identity test with iterable for int failed %v", err)
	}
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenIterI8 := func(a *QCDenseI8) bool {
		var a1, ret, correct *Dense
		var identity int8
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI8, nil); err != nil {
		t.Errorf("Identity test with iterable for int8 failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenIterI16 := func(a *QCDenseI16) bool {
		var a1, ret, correct *Dense
		var identity int16
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI16, nil); err != nil {
		t.Errorf("Identity test with iterable for int16 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenIterI32 := func(a *QCDenseI32) bool {
		var a1, ret, correct *Dense
		var identity int32
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI32, nil); err != nil {
		t.Errorf("Identity test with iterable for int32 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenIterI64 := func(a *QCDenseI64) bool {
		var a1, ret, correct *Dense
		var identity int64
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI64, nil); err != nil {
		t.Errorf("Identity test with iterable for int64 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenIterU := func(a *QCDenseU) bool {
		var a1, ret, correct *Dense
		var identity uint
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU, nil); err != nil {
		t.Errorf("Identity test with iterable for uint failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenIterU8 := func(a *QCDenseU8) bool {
		var a1, ret, correct *Dense
		var identity uint8
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU8, nil); err != nil {
		t.Errorf("Identity test with iterable for uint8 failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenIterU16 := func(a *QCDenseU16) bool {
		var a1, ret, correct *Dense
		var identity uint16
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU16, nil); err != nil {
		t.Errorf("Identity test with iterable for uint16 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenIterU32 := func(a *QCDenseU32) bool {
		var a1, ret, correct *Dense
		var identity uint32
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU32, nil); err != nil {
		t.Errorf("Identity test with iterable for uint32 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenIterU64 := func(a *QCDenseU64) bool {
		var a1, ret, correct *Dense
		var identity uint64
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU64, nil); err != nil {
		t.Errorf("Identity test with iterable for uint64 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenIterF32 := func(a *QCDenseF32) bool {
		var a1, ret, correct *Dense
		var identity float32
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF32, nil); err != nil {
		t.Errorf("Identity test with iterable for float32 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenIterF64 := func(a *QCDenseF64) bool {
		var a1, ret, correct *Dense
		var identity float64
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF64, nil); err != nil {
		t.Errorf("Identity test with iterable for float64 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenIterC64 := func(a *QCDenseC64) bool {
		var a1, ret, correct *Dense
		var identity complex64
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC64, nil); err != nil {
		t.Errorf("Identity test with iterable for complex64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenIterC128 := func(a *QCDenseC128) bool {
		var a1, ret, correct *Dense
		var identity complex128
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Trans(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Trans(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC128, nil); err != nil {
		t.Errorf("Identity test with iterable for complex128 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.Trans(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Trans(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenIterI := func(a *QCDenseI) bool {
		var a1, ret, correct *Dense
		var identity int
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI, nil); err != nil {
		t.Errorf("Identity test with iterable for int failed %v", err)
	}
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenIterI8 := func(a *QCDenseI8) bool {
		var a1, ret, correct *Dense
		var identity int8
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI8, nil); err != nil {
		t.Errorf("Identity test with iterable for int8 failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenIterI16 := func(a *QCDenseI16) bool {
		var a1, ret, correct *Dense
		var identity int16
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI16, nil); err != nil {
		t.Errorf("Identity test with iterable for int16 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenIterI32 := func(a *QCDenseI32) bool {
		var a1, ret, correct *Dense
		var identity int32
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI32, nil); err != nil {
		t.Errorf("Identity test with iterable for int32 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenIterI64 := func(a *QCDenseI64) bool {
		var a1, ret, correct *Dense
		var identity int64
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI64, nil); err != nil {
		t.Errorf("Identity test with iterable for int64 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenIterU := func(a *QCDenseU) bool {
		var a1, ret, correct *Dense
		var identity uint
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU, nil); err != nil {
		t.Errorf("Identity test with iterable for uint failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenIterU8 := func(a *QCDenseU8) bool {
		var a1, ret, correct *Dense
		var identity uint8
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU8, nil); err != nil {
		t.Errorf("Identity test with iterable for uint8 failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenIterU16 := func(a *QCDenseU16) bool {
		var a1, ret, correct *Dense
		var identity uint16
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU16, nil); err != nil {
		t.Errorf("Identity test with iterable for uint16 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenIterU32 := func(a *QCDenseU32) bool {
		var a1, ret, correct *Dense
		var identity uint32
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU32, nil); err != nil {
		t.Errorf("Identity test with iterable for uint32 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenIterU64 := func(a *QCDenseU64) bool {
		var a1, ret, correct *Dense
		var identity uint64
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU64, nil); err != nil {
		t.Errorf("Identity test with iterable for uint64 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenIterF32 := func(a *QCDenseF32) bool {
		var a1, ret, correct *Dense
		var identity float32
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF32, nil); err != nil {
		t.Errorf("Identity test with iterable for float32 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenIterF64 := func(a *QCDenseF64) bool {
		var a1, ret, correct *Dense
		var identity float64
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF64, nil); err != nil {
		t.Errorf("Identity test with iterable for float64 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenIterC64 := func(a *QCDenseC64) bool {
		var a1, ret, correct *Dense
		var identity complex64
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC64, nil); err != nil {
		t.Errorf("Identity test with iterable for complex64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenIterC128 := func(a *QCDenseC128) bool {
		var a1, ret, correct *Dense
		var identity complex128
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.TransInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.TransInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC128, nil); err != nil {
		t.Errorf("Identity test with iterable for complex128 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.TransInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
}

/* TransInvR */

func TestTransInvRBasicProperties(t *testing.T) {
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.TransInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.TransInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
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

	idenIterI := func(a *QCDenseI) bool {
		var a1, ret, correct *Dense
		var identity int
		identity = 1
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI, nil); err != nil {
		t.Errorf("Identity test with iterable for int failed %v", err)
	}
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenIterI8 := func(a *QCDenseI8) bool {
		var a1, ret, correct *Dense
		var identity int8
		identity = 1
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI8, nil); err != nil {
		t.Errorf("Identity test with iterable for int8 failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenIterI16 := func(a *QCDenseI16) bool {
		var a1, ret, correct *Dense
		var identity int16
		identity = 1
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI16, nil); err != nil {
		t.Errorf("Identity test with iterable for int16 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenIterI32 := func(a *QCDenseI32) bool {
		var a1, ret, correct *Dense
		var identity int32
		identity = 1
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI32, nil); err != nil {
		t.Errorf("Identity test with iterable for int32 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenIterI64 := func(a *QCDenseI64) bool {
		var a1, ret, correct *Dense
		var identity int64
		identity = 1
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI64, nil); err != nil {
		t.Errorf("Identity test with iterable for int64 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenIterU := func(a *QCDenseU) bool {
		var a1, ret, correct *Dense
		var identity uint
		identity = 1
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU, nil); err != nil {
		t.Errorf("Identity test with iterable for uint failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenIterU8 := func(a *QCDenseU8) bool {
		var a1, ret, correct *Dense
		var identity uint8
		identity = 1
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU8, nil); err != nil {
		t.Errorf("Identity test with iterable for uint8 failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenIterU16 := func(a *QCDenseU16) bool {
		var a1, ret, correct *Dense
		var identity uint16
		identity = 1
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU16, nil); err != nil {
		t.Errorf("Identity test with iterable for uint16 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenIterU32 := func(a *QCDenseU32) bool {
		var a1, ret, correct *Dense
		var identity uint32
		identity = 1
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU32, nil); err != nil {
		t.Errorf("Identity test with iterable for uint32 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenIterU64 := func(a *QCDenseU64) bool {
		var a1, ret, correct *Dense
		var identity uint64
		identity = 1
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU64, nil); err != nil {
		t.Errorf("Identity test with iterable for uint64 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenIterF32 := func(a *QCDenseF32) bool {
		var a1, ret, correct *Dense
		var identity float32
		identity = 1
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF32, nil); err != nil {
		t.Errorf("Identity test with iterable for float32 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenIterF64 := func(a *QCDenseF64) bool {
		var a1, ret, correct *Dense
		var identity float64
		identity = 1
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF64, nil); err != nil {
		t.Errorf("Identity test with iterable for float64 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenIterC64 := func(a *QCDenseC64) bool {
		var a1, ret, correct *Dense
		var identity complex64
		identity = 1
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC64, nil); err != nil {
		t.Errorf("Identity test with iterable for complex64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenIterC128 := func(a *QCDenseC128) bool {
		var a1, ret, correct *Dense
		var identity complex128
		identity = 1
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.Scale(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.Scale(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC128, nil); err != nil {
		t.Errorf("Identity test with iterable for complex128 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.Scale(b)
		correct, _ := incr.Add(ret)

		check, _ := a.Scale(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
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

	idenIterI := func(a *QCDenseI) bool {
		var a1, ret, correct *Dense
		var identity int
		identity = 1
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI, nil); err != nil {
		t.Errorf("Identity test with iterable for int failed %v", err)
	}
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	idenIterI8 := func(a *QCDenseI8) bool {
		var a1, ret, correct *Dense
		var identity int8
		identity = 1
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI8, nil); err != nil {
		t.Errorf("Identity test with iterable for int8 failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	idenIterI16 := func(a *QCDenseI16) bool {
		var a1, ret, correct *Dense
		var identity int16
		identity = 1
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI16, nil); err != nil {
		t.Errorf("Identity test with iterable for int16 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	idenIterI32 := func(a *QCDenseI32) bool {
		var a1, ret, correct *Dense
		var identity int32
		identity = 1
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI32, nil); err != nil {
		t.Errorf("Identity test with iterable for int32 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	idenIterI64 := func(a *QCDenseI64) bool {
		var a1, ret, correct *Dense
		var identity int64
		identity = 1
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterI64, nil); err != nil {
		t.Errorf("Identity test with iterable for int64 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	idenIterU := func(a *QCDenseU) bool {
		var a1, ret, correct *Dense
		var identity uint
		identity = 1
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU, nil); err != nil {
		t.Errorf("Identity test with iterable for uint failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	idenIterU8 := func(a *QCDenseU8) bool {
		var a1, ret, correct *Dense
		var identity uint8
		identity = 1
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU8, nil); err != nil {
		t.Errorf("Identity test with iterable for uint8 failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	idenIterU16 := func(a *QCDenseU16) bool {
		var a1, ret, correct *Dense
		var identity uint16
		identity = 1
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU16, nil); err != nil {
		t.Errorf("Identity test with iterable for uint16 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	idenIterU32 := func(a *QCDenseU32) bool {
		var a1, ret, correct *Dense
		var identity uint32
		identity = 1
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU32, nil); err != nil {
		t.Errorf("Identity test with iterable for uint32 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	idenIterU64 := func(a *QCDenseU64) bool {
		var a1, ret, correct *Dense
		var identity uint64
		identity = 1
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterU64, nil); err != nil {
		t.Errorf("Identity test with iterable for uint64 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	idenIterF32 := func(a *QCDenseF32) bool {
		var a1, ret, correct *Dense
		var identity float32
		identity = 1
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF32, nil); err != nil {
		t.Errorf("Identity test with iterable for float32 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	idenIterF64 := func(a *QCDenseF64) bool {
		var a1, ret, correct *Dense
		var identity float64
		identity = 1
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterF64, nil); err != nil {
		t.Errorf("Identity test with iterable for float64 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	idenIterC64 := func(a *QCDenseC64) bool {
		var a1, ret, correct *Dense
		var identity complex64
		identity = 1
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC64, nil); err != nil {
		t.Errorf("Identity test with iterable for complex64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	idenIterC128 := func(a *QCDenseC128) bool {
		var a1, ret, correct *Dense
		var identity complex128
		identity = 1
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)
		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.ScaleInv(identity, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe:
		ret, _ = a1.ScaleInv(identity)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		return true
	}
	if err := quick.Check(idenIterC128, nil); err != nil {
		t.Errorf("Identity test with iterable for complex128 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.ScaleInv(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInv(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
}

/* ScaleInvR */

func TestScaleInvRBasicProperties(t *testing.T) {
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.ScaleInvR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.ScaleInvR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
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

	pow0IterI := func(a *QCDenseI) bool {
		var a1, ret, correct *Dense
		var zero int
		correct = newDense(Int, a.len())
		correct.Memset(int(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterI, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
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

	pow0IterI8 := func(a *QCDenseI8) bool {
		var a1, ret, correct *Dense
		var zero int8
		correct = newDense(Int8, a.len())
		correct.Memset(int8(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterI8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
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

	pow0IterI16 := func(a *QCDenseI16) bool {
		var a1, ret, correct *Dense
		var zero int16
		correct = newDense(Int16, a.len())
		correct.Memset(int16(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterI16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
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

	pow0IterI32 := func(a *QCDenseI32) bool {
		var a1, ret, correct *Dense
		var zero int32
		correct = newDense(Int32, a.len())
		correct.Memset(int32(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterI32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
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

	pow0IterI64 := func(a *QCDenseI64) bool {
		var a1, ret, correct *Dense
		var zero int64
		correct = newDense(Int64, a.len())
		correct.Memset(int64(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterI64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
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

	pow0IterU := func(a *QCDenseU) bool {
		var a1, ret, correct *Dense
		var zero uint
		correct = newDense(Uint, a.len())
		correct.Memset(uint(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterU, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
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

	pow0IterU8 := func(a *QCDenseU8) bool {
		var a1, ret, correct *Dense
		var zero uint8
		correct = newDense(Uint8, a.len())
		correct.Memset(uint8(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterU8, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
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

	pow0IterU16 := func(a *QCDenseU16) bool {
		var a1, ret, correct *Dense
		var zero uint16
		correct = newDense(Uint16, a.len())
		correct.Memset(uint16(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterU16, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
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

	pow0IterU32 := func(a *QCDenseU32) bool {
		var a1, ret, correct *Dense
		var zero uint32
		correct = newDense(Uint32, a.len())
		correct.Memset(uint32(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterU32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
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

	pow0IterU64 := func(a *QCDenseU64) bool {
		var a1, ret, correct *Dense
		var zero uint64
		correct = newDense(Uint64, a.len())
		correct.Memset(uint64(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterU64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
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

	pow0IterF32 := func(a *QCDenseF32) bool {
		var a1, ret, correct *Dense
		var zero float32
		correct = newDense(Float32, a.len())
		correct.Memset(float32(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterF32, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
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

	pow0IterF64 := func(a *QCDenseF64) bool {
		var a1, ret, correct *Dense
		var zero float64
		correct = newDense(Float64, a.len())
		correct.Memset(float64(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterF64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
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

	pow0IterC64 := func(a *QCDenseC64) bool {
		var a1, ret, correct *Dense
		var zero complex64
		correct = newDense(Complex64, a.len())
		correct.Memset(complex64(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterC64, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
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

	pow0IterC128 := func(a *QCDenseC128) bool {
		var a1, ret, correct *Dense
		var zero complex128
		correct = newDense(Complex128, a.len())
		correct.Memset(complex128(1))

		correct, _ = sliceDense(correct, makeRS(0, 5))
		correct = correct.Materialize().(*Dense)

		a1, _ = sliceDense(a.Dense, makeRS(0, 5))
		ret, _ = a1.PowOf(zero, UseUnsafe())
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}

		// safe
		ret, _ = a1.PowOf(zero)
		if !allClose(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(pow0IterC128, nil); err != nil {
		t.Errorf("Pow 0 failed")
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.PowOf(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOf(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
}

/* PowOfR */

func TestPowOfRBasicProperties(t *testing.T) {
	incrI := func(a, incr *QCDenseI, b int) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int)[0:10], check.Data().([]int)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI, nil); err != nil {
		t.Error("Incr function test for int failed %v", err)
	}
	incrI8 := func(a, incr *QCDenseI8, b int8) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int8)[0:10], check.Data().([]int8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI8, nil); err != nil {
		t.Error("Incr function test for int8 failed %v", err)
	}
	incrI16 := func(a, incr *QCDenseI16, b int16) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int16)[0:10], check.Data().([]int16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI16, nil); err != nil {
		t.Error("Incr function test for int16 failed %v", err)
	}
	incrI32 := func(a, incr *QCDenseI32, b int32) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int32)[0:10], check.Data().([]int32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI32, nil); err != nil {
		t.Error("Incr function test for int32 failed %v", err)
	}
	incrI64 := func(a, incr *QCDenseI64, b int64) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]int64)[0:10], check.Data().([]int64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrI64, nil); err != nil {
		t.Error("Incr function test for int64 failed %v", err)
	}
	incrU := func(a, incr *QCDenseU, b uint) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint)[0:10], check.Data().([]uint)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU, nil); err != nil {
		t.Error("Incr function test for uint failed %v", err)
	}
	incrU8 := func(a, incr *QCDenseU8, b uint8) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint8)[0:10], check.Data().([]uint8)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU8, nil); err != nil {
		t.Error("Incr function test for uint8 failed %v", err)
	}
	incrU16 := func(a, incr *QCDenseU16, b uint16) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint16)[0:10], check.Data().([]uint16)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU16, nil); err != nil {
		t.Error("Incr function test for uint16 failed %v", err)
	}
	incrU32 := func(a, incr *QCDenseU32, b uint32) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint32)[0:10], check.Data().([]uint32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU32, nil); err != nil {
		t.Error("Incr function test for uint32 failed %v", err)
	}
	incrU64 := func(a, incr *QCDenseU64, b uint64) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]uint64)[0:10], check.Data().([]uint64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrU64, nil); err != nil {
		t.Error("Incr function test for uint64 failed %v", err)
	}
	incrF32 := func(a, incr *QCDenseF32, b float32) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float32)[0:10], check.Data().([]float32)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF32, nil); err != nil {
		t.Error("Incr function test for float32 failed %v", err)
	}
	incrF64 := func(a, incr *QCDenseF64, b float64) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]float64)[0:10], check.Data().([]float64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrF64, nil); err != nil {
		t.Error("Incr function test for float64 failed %v", err)
	}
	incrC64 := func(a, incr *QCDenseC64, b complex64) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex64)[0:10], check.Data().([]complex64)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC64, nil); err != nil {
		t.Error("Incr function test for complex64 failed %v", err)
	}
	incrC128 := func(a, incr *QCDenseC128, b complex128) bool {
		// build correct
		ret, _ := a.PowOfR(b)
		correct, _ := incr.Add(ret)

		check, _ := a.PowOfR(b, WithIncr(incr.Dense))
		if check != incr.Dense {
			t.Error("Expected incr.Dense == check")
			return false
		}
		if !allClose(correct.Data(), check.Data()) {
			t.Errorf("Correct: %v, check %v", correct.Data().([]complex128)[0:10], check.Data().([]complex128)[0:10])
			return false
		}

		return true
	}
	if err := quick.Check(incrC128, nil); err != nil {
		t.Error("Incr function test for complex128 failed %v", err)
	}
}
