package tensor

import (
	"reflect"
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestAdditionBasicProperties(t *testing.T) {
	idenI := func(a *QCDenseI) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI, nil); err != nil {
		t.Error(err)
	}
	commI := func(a, b *QCDenseI) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI, nil); err != nil {
		t.Error(err)
	}
	assocI := func(a, b, c *QCDenseI) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocI, nil); err != nil {
		t.Error(err)
	}
	idenI8 := func(a *QCDenseI8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int8, a.len())
		correct = newDense(Int8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI8, nil); err != nil {
		t.Error(err)
	}
	commI8 := func(a, b *QCDenseI8) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI8, nil); err != nil {
		t.Error(err)
	}
	assocI8 := func(a, b, c *QCDenseI8) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocI8, nil); err != nil {
		t.Error(err)
	}
	idenI16 := func(a *QCDenseI16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int16, a.len())
		correct = newDense(Int16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI16, nil); err != nil {
		t.Error(err)
	}
	commI16 := func(a, b *QCDenseI16) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI16, nil); err != nil {
		t.Error(err)
	}
	assocI16 := func(a, b, c *QCDenseI16) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocI16, nil); err != nil {
		t.Error(err)
	}
	idenI32 := func(a *QCDenseI32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int32, a.len())
		correct = newDense(Int32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI32, nil); err != nil {
		t.Error(err)
	}
	commI32 := func(a, b *QCDenseI32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI32, nil); err != nil {
		t.Error(err)
	}
	assocI32 := func(a, b, c *QCDenseI32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocI32, nil); err != nil {
		t.Error(err)
	}
	idenI64 := func(a *QCDenseI64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Int64, a.len())
		correct = newDense(Int64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenI64, nil); err != nil {
		t.Error(err)
	}
	commI64 := func(a, b *QCDenseI64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commI64, nil); err != nil {
		t.Error(err)
	}
	assocI64 := func(a, b, c *QCDenseI64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocI64, nil); err != nil {
		t.Error(err)
	}
	idenU := func(a *QCDenseU) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint, a.len())
		correct = newDense(Uint, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU, nil); err != nil {
		t.Error(err)
	}
	commU := func(a, b *QCDenseU) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU, nil); err != nil {
		t.Error(err)
	}
	assocU := func(a, b, c *QCDenseU) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocU, nil); err != nil {
		t.Error(err)
	}
	idenU8 := func(a *QCDenseU8) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint8, a.len())
		correct = newDense(Uint8, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU8, nil); err != nil {
		t.Error(err)
	}
	commU8 := func(a, b *QCDenseU8) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU8, nil); err != nil {
		t.Error(err)
	}
	assocU8 := func(a, b, c *QCDenseU8) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocU8, nil); err != nil {
		t.Error(err)
	}
	idenU16 := func(a *QCDenseU16) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint16, a.len())
		correct = newDense(Uint16, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU16, nil); err != nil {
		t.Error(err)
	}
	commU16 := func(a, b *QCDenseU16) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU16, nil); err != nil {
		t.Error(err)
	}
	assocU16 := func(a, b, c *QCDenseU16) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocU16, nil); err != nil {
		t.Error(err)
	}
	idenU32 := func(a *QCDenseU32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint32, a.len())
		correct = newDense(Uint32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU32, nil); err != nil {
		t.Error(err)
	}
	commU32 := func(a, b *QCDenseU32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU32, nil); err != nil {
		t.Error(err)
	}
	assocU32 := func(a, b, c *QCDenseU32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocU32, nil); err != nil {
		t.Error(err)
	}
	idenU64 := func(a *QCDenseU64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Uint64, a.len())
		correct = newDense(Uint64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenU64, nil); err != nil {
		t.Error(err)
	}
	commU64 := func(a, b *QCDenseU64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commU64, nil); err != nil {
		t.Error(err)
	}
	assocU64 := func(a, b, c *QCDenseU64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocU64, nil); err != nil {
		t.Error(err)
	}
	idenF32 := func(a *QCDenseF32) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float32, a.len())
		correct = newDense(Float32, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF32, nil); err != nil {
		t.Error(err)
	}
	commF32 := func(a, b *QCDenseF32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF32, nil); err != nil {
		t.Error(err)
	}
	assocF32 := func(a, b, c *QCDenseF32) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocF32, nil); err != nil {
		t.Error(err)
	}
	idenF64 := func(a *QCDenseF64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Float64, a.len())
		correct = newDense(Float64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenF64, nil); err != nil {
		t.Error(err)
	}
	commF64 := func(a, b *QCDenseF64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commF64, nil); err != nil {
		t.Error(err)
	}
	assocF64 := func(a, b, c *QCDenseF64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocF64, nil); err != nil {
		t.Error(err)
	}
	idenC64 := func(a *QCDenseC64) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex64, a.len())
		correct = newDense(Complex64, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC64, nil); err != nil {
		t.Error(err)
	}
	commC64 := func(a, b *QCDenseC64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC64, nil); err != nil {
		t.Error(err)
	}
	assocC64 := func(a, b, c *QCDenseC64) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocC64, nil); err != nil {
		t.Error(err)
	}
	idenC128 := func(a *QCDenseC128) bool {
		var ret, correct, identity *Dense
		identity = newDense(Complex128, a.len())
		correct = newDense(Complex128, a.len())
		copyDense(correct, a.Dense)

		ret, _ = identity.Add(a.Dense)
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(idenC128, nil); err != nil {
		t.Error(err)
	}
	commC128 := func(a, b *QCDenseC128) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret2, _ = b.Add(a.Dense)
		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(commC128, nil); err != nil {
		t.Error(err)
	}
	assocC128 := func(a, b, c *QCDenseC128) bool {
		var ret1, ret2 *Dense
		ret1, _ = a.Add(b.Dense)
		ret1, _ = ret1.Add(c.Dense)

		ret2, _ = b.Add(c.Dense)
		ret2, _ = a.Add(ret2)

		if !reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Errorf("%v\n%v", ret1.Data(), ret2.Data())
			return false
		}
		return true
	}
	if err := quick.Check(assocC128, nil); err != nil {
		t.Error(err)
	}
}
func TestAdditionFuncOpts(t *testing.T) {
	var f func(*QCDenseI) bool
	f = func(a *QCDenseI) bool {
		identity := newDense(Int, a.len()+1)
		if _, err := identity.Add(a.Dense); err == nil {
			t.Error("Failed length mismatch test")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}

	// safe
	f = func(a *QCDenseI) bool {
		var identity, ret *Dense
		var err error
		identity = newDense(Int, a.len())
		if ret, err = identity.Add(a.Dense); err != nil {
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
		t.Error(err)
	}

	// reuse
	f = func(a *QCDenseI) bool {
		var identity, ret, correct, reuse *Dense
		var err error
		identity = newDense(Int, a.len())
		reuse = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)
		if ret, err = identity.Add(a.Dense, WithReuse(reuse)); err != nil {
			t.Error(err)
			return false
		}
		if ret != reuse {
			t.Error("Expected ret == reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}

		// wrong reuse type
		reuse = newDense(Bool, a.len())
		if _, err = identity.Add(a.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when using a reuse with a type mismatch")
			return false
		}

		// wrong reuse length
		reuse = newDense(Int, a.len()+1)
		if _, err = identity.Add(a.Dense, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when using a reuse with a size mismatch")
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}

	// unsafe
	f = func(a *QCDenseI) bool {
		var identity, ret, correct *Dense
		var err error
		identity = newDense(Int, a.len())
		correct = newDense(Int, a.len())
		copyDense(correct, a.Dense)

		if ret, err = identity.Add(a.Dense, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != identity {
			t.Error("Expected ret == reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			return false
		}
		return true
	}
	if err := quick.Check(f, nil); err != nil {
		t.Error(err)
	}
}

/* int */

func TestSubtractionI(t *testing.T) {
	var f func(*QCDenseI) bool
	var err error

	f = func(x *QCDenseI) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int, x.len())
		correct = newDense(Int, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Int, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Int, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseI) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero int
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Int, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Int, x.len())
		reuse2 = newDense(Int, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Int, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationI(t *testing.T) {
	var f func(*QCDenseI) bool
	var err error

	f = func(x *QCDenseI) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar int = 1
		one = newDense(Int, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Int, x.len())
		one.Memset(oneScalar)
		correct = newDense(Int, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Int, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI) bool {
		var ret, correct, reuse *Dense
		var one int
		var err error
		one = 1
		correct = newDense(Int, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Int, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* int8 */

func TestSubtractionI8(t *testing.T) {
	var f func(*QCDenseI8) bool
	var err error

	f = func(x *QCDenseI8) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int8, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int8, x.len())
		correct = newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int8, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Int8, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Int8, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseI8) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero int8
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Int8, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Int8, x.len())
		reuse2 = newDense(Int8, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Int8, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationI8(t *testing.T) {
	var f func(*QCDenseI8) bool
	var err error

	f = func(x *QCDenseI8) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar int8 = 1
		one = newDense(Int8, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Int8, x.len())
		one.Memset(oneScalar)
		correct = newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int8, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int8, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Int8, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI8) bool {
		var ret, correct, reuse *Dense
		var one int8
		var err error
		one = 1
		correct = newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Int8, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* int16 */

func TestSubtractionI16(t *testing.T) {
	var f func(*QCDenseI16) bool
	var err error

	f = func(x *QCDenseI16) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int16, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int16, x.len())
		correct = newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int16, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Int16, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Int16, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseI16) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero int16
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Int16, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Int16, x.len())
		reuse2 = newDense(Int16, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Int16, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationI16(t *testing.T) {
	var f func(*QCDenseI16) bool
	var err error

	f = func(x *QCDenseI16) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar int16 = 1
		one = newDense(Int16, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Int16, x.len())
		one.Memset(oneScalar)
		correct = newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int16, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int16, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Int16, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI16) bool {
		var ret, correct, reuse *Dense
		var one int16
		var err error
		one = 1
		correct = newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Int16, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* int32 */

func TestSubtractionI32(t *testing.T) {
	var f func(*QCDenseI32) bool
	var err error

	f = func(x *QCDenseI32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int32, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int32, x.len())
		correct = newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int32, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Int32, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Int32, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseI32) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero int32
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Int32, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Int32, x.len())
		reuse2 = newDense(Int32, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Int32, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationI32(t *testing.T) {
	var f func(*QCDenseI32) bool
	var err error

	f = func(x *QCDenseI32) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar int32 = 1
		one = newDense(Int32, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Int32, x.len())
		one.Memset(oneScalar)
		correct = newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int32, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int32, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Int32, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI32) bool {
		var ret, correct, reuse *Dense
		var one int32
		var err error
		one = 1
		correct = newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Int32, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* int64 */

func TestSubtractionI64(t *testing.T) {
	var f func(*QCDenseI64) bool
	var err error

	f = func(x *QCDenseI64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int64, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int64, x.len())
		correct = newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int64, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Int64, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Int64, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseI64) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero int64
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Int64, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Int64, x.len())
		reuse2 = newDense(Int64, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Int64, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationI64(t *testing.T) {
	var f func(*QCDenseI64) bool
	var err error

	f = func(x *QCDenseI64) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar int64 = 1
		one = newDense(Int64, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Int64, x.len())
		one.Memset(oneScalar)
		correct = newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int64, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int64, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Int64, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI64) bool {
		var ret, correct, reuse *Dense
		var one int64
		var err error
		one = 1
		correct = newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Int64, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* uint */

func TestSubtractionU(t *testing.T) {
	var f func(*QCDenseU) bool
	var err error

	f = func(x *QCDenseU) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint, x.len())
		correct = newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Uint, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Uint, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseU) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero uint
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Uint, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Uint, x.len())
		reuse2 = newDense(Uint, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Uint, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationU(t *testing.T) {
	var f func(*QCDenseU) bool
	var err error

	f = func(x *QCDenseU) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar uint = 1
		one = newDense(Uint, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Uint, x.len())
		one.Memset(oneScalar)
		correct = newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Uint, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU) bool {
		var ret, correct, reuse *Dense
		var one uint
		var err error
		one = 1
		correct = newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Uint, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* uint8 */

func TestSubtractionU8(t *testing.T) {
	var f func(*QCDenseU8) bool
	var err error

	f = func(x *QCDenseU8) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint8, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint8, x.len())
		correct = newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint8, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Uint8, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Uint8, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseU8) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero uint8
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Uint8, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Uint8, x.len())
		reuse2 = newDense(Uint8, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Uint8, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationU8(t *testing.T) {
	var f func(*QCDenseU8) bool
	var err error

	f = func(x *QCDenseU8) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar uint8 = 1
		one = newDense(Uint8, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Uint8, x.len())
		one.Memset(oneScalar)
		correct = newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint8, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint8, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Uint8, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU8) bool {
		var ret, correct, reuse *Dense
		var one uint8
		var err error
		one = 1
		correct = newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Uint8, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* uint16 */

func TestSubtractionU16(t *testing.T) {
	var f func(*QCDenseU16) bool
	var err error

	f = func(x *QCDenseU16) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint16, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint16, x.len())
		correct = newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint16, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Uint16, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Uint16, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseU16) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero uint16
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Uint16, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Uint16, x.len())
		reuse2 = newDense(Uint16, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Uint16, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationU16(t *testing.T) {
	var f func(*QCDenseU16) bool
	var err error

	f = func(x *QCDenseU16) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar uint16 = 1
		one = newDense(Uint16, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Uint16, x.len())
		one.Memset(oneScalar)
		correct = newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint16, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint16, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Uint16, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU16) bool {
		var ret, correct, reuse *Dense
		var one uint16
		var err error
		one = 1
		correct = newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Uint16, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* uint32 */

func TestSubtractionU32(t *testing.T) {
	var f func(*QCDenseU32) bool
	var err error

	f = func(x *QCDenseU32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint32, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint32, x.len())
		correct = newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint32, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Uint32, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Uint32, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseU32) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero uint32
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Uint32, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Uint32, x.len())
		reuse2 = newDense(Uint32, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Uint32, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationU32(t *testing.T) {
	var f func(*QCDenseU32) bool
	var err error

	f = func(x *QCDenseU32) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar uint32 = 1
		one = newDense(Uint32, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Uint32, x.len())
		one.Memset(oneScalar)
		correct = newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint32, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint32, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Uint32, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU32) bool {
		var ret, correct, reuse *Dense
		var one uint32
		var err error
		one = 1
		correct = newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Uint32, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* uint64 */

func TestSubtractionU64(t *testing.T) {
	var f func(*QCDenseU64) bool
	var err error

	f = func(x *QCDenseU64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint64, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint64, x.len())
		correct = newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint64, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Uint64, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Uint64, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseU64) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero uint64
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Uint64, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Uint64, x.len())
		reuse2 = newDense(Uint64, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Uint64, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationU64(t *testing.T) {
	var f func(*QCDenseU64) bool
	var err error

	f = func(x *QCDenseU64) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar uint64 = 1
		one = newDense(Uint64, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Uint64, x.len())
		one.Memset(oneScalar)
		correct = newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint64, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint64, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Uint64, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU64) bool {
		var ret, correct, reuse *Dense
		var one uint64
		var err error
		one = 1
		correct = newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Uint64, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* float32 */

func TestSubtractionF32(t *testing.T) {
	var f func(*QCDenseF32) bool
	var err error

	f = func(x *QCDenseF32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Float32, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Float32, x.len())
		correct = newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float32, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Float32, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Float32, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseF32) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero float32
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Float32, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Float32, x.len())
		reuse2 = newDense(Float32, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Float32, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationF32(t *testing.T) {
	var f func(*QCDenseF32) bool
	var err error

	f = func(x *QCDenseF32) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar float32 = 1
		one = newDense(Float32, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Float32, x.len())
		one.Memset(oneScalar)
		correct = newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float32, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Float32, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Float32, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseF32) bool {
		var ret, correct, reuse *Dense
		var one float32
		var err error
		one = 1
		correct = newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Float32, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* float64 */

func TestSubtractionF64(t *testing.T) {
	var f func(*QCDenseF64) bool
	var err error

	f = func(x *QCDenseF64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Float64, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Float64, x.len())
		correct = newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float64, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Float64, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Float64, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseF64) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero float64
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Float64, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Float64, x.len())
		reuse2 = newDense(Float64, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Float64, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationF64(t *testing.T) {
	var f func(*QCDenseF64) bool
	var err error

	f = func(x *QCDenseF64) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar float64 = 1
		one = newDense(Float64, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Float64, x.len())
		one.Memset(oneScalar)
		correct = newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float64, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Float64, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Float64, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseF64) bool {
		var ret, correct, reuse *Dense
		var one float64
		var err error
		one = 1
		correct = newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Float64, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* complex64 */

func TestSubtractionC64(t *testing.T) {
	var f func(*QCDenseC64) bool
	var err error

	f = func(x *QCDenseC64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Complex64, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Complex64, x.len())
		correct = newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex64, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Complex64, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Complex64, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseC64) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero complex64
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Complex64, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Complex64, x.len())
		reuse2 = newDense(Complex64, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Complex64, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationC64(t *testing.T) {
	var f func(*QCDenseC64) bool
	var err error

	f = func(x *QCDenseC64) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar complex64 = 1
		one = newDense(Complex64, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Complex64, x.len())
		one.Memset(oneScalar)
		correct = newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex64, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Complex64, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Complex64, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseC64) bool {
		var ret, correct, reuse *Dense
		var one complex64
		var err error
		one = 1
		correct = newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Complex64, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}

/* complex128 */

func TestSubtractionC128(t *testing.T) {
	var f func(*QCDenseC128) bool
	var err error

	f = func(x *QCDenseC128) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Complex128, x.len()+1)

		if _, err = x.Sub(zero); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Complex128, x.len())
		correct = newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		if ret, err = x.Sub(zero); err != nil {
			t.Errorf("Failed to minus 0 :%v", err)
			return false
		}
		if ret == zero || ret == x.Dense {
			t.Error("Expected safe identity function to return a completely new *Dense")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Safe Identity Test failed. \nWant:%v\nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex128, x.len()+1)
		if _, err = x.Sub(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		reuse = newDense(Complex128, x.len())
		if ret, err = x.Sub(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed resue identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = x.Sub(zero, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != ret {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// subtractions are NOT commutative
		zero = newDense(Complex128, x.len())
		ret, _ = x.Sub(zero)
		ret2, _ := zero.Sub(x.Dense)
		if reflect.DeepEqual(ret.Data(), ret2.Data()) {
			t.Error("Subtraction should NOT be commutative")
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Identity test: %v", err)
	}

	f = func(x *QCDenseC128) bool {
		var ret1, ret2, reuse1, reuse2 *Dense
		var zero complex128
		var err error
		zero = 0

		// safe TransInv/TransInvR tests
		if ret1, err = x.TransInv(zero); err != nil {
			t.Errorf("Error while safe TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero); err != nil {
			t.Errorf("Error while safe TransInvR: %v", err)
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// reuse - wrong length
		reuse1 = newDense(Complex128, x.len()+1)
		if _, err = x.TransInv(zero, WithReuse(reuse1)); err == nil {
			t.Error("Expected an error when reuse size does not match")
			return false
		}

		// reuse - correct
		reuse1 = newDense(Complex128, x.len())
		reuse2 = newDense(Complex128, x.len())
		if ret1, err = x.TransInv(zero, WithReuse(reuse1)); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x.TransInvR(zero, WithReuse(reuse2)); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != reuse1 {
			t.Errorf("Expected ret1 to be the same as reuse1")
			return false
		}
		if ret2 != reuse2 {
			t.Errorf("Expected ret2 to be the same as reuse2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}

		// unsafe
		x2 := newDense(Complex128, x.len())
		copyDense(x2, x.Dense)
		if ret1, err = x.TransInv(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInv: %v", err)
			return false
		}
		if ret2, err = x2.TransInvR(zero, UseUnsafe()); err != nil {
			t.Errorf("Error while reuse TransInvR: %v", err)
			return false
		}
		if ret1 != x.Dense {
			t.Errorf("Expected ret1 to be the same as x")
			return false
		}
		if ret2 != x2 {
			t.Errorf("Expected ret2 to be the same as x2")
			return false
		}
		if reflect.DeepEqual(ret1.Data(), ret2.Data()) {
			t.Error("Expected subtraction to be non commutative")
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed TransInv test: %v", err)
	}
}
func TestMultiplicationC128(t *testing.T) {
	var f func(*QCDenseC128) bool
	var err error

	f = func(x *QCDenseC128) bool {
		var ret, correct, one, reuse *Dense
		var err error
		var oneScalar complex128 = 1
		one = newDense(Complex128, x.len()+1)
		one.Memset(oneScalar)

		// basic length test
		if _, err = one.Mul(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		one = newDense(Complex128, x.len())
		one.Memset(oneScalar)
		correct = newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		if ret, err = one.Mul(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == one || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex128, x.len()+1)
		if _, err = x.Mul(one, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Complex128, x.len())
		if ret, err = one.Mul(x.Dense, WithReuse(reuse)); err != nil {
			t.Errorf("Failed reuse identity test: %v", err)
			return false
		}
		if ret != reuse {
			t.Errorf("Failed reuse identity test. Expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed reuse identity test: Operation incorrect: \nWant: %v \nGot: %v", correct.Data(), ret.Data())
		}

		// unsafe identity
		if ret, err = one.Mul(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != one {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		one = newDense(Complex128, x.len())
		one.Memset(oneScalar)
		if ret, err = x.Mul(one); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed commutativity/associativity  test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseC128) bool {
		var ret, correct, reuse *Dense
		var one complex128
		var err error
		one = 1
		correct = newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		// Safe Scale
		if ret, err = x.Scale(one); err != nil {
			t.Errorf("Failed Safe Scale test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Scale: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Scale
		reuse = newDense(Complex128, x.len())
		if ret, err = x.Scale(one, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Scale test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Scale
		if ret, err = x.Scale(one, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Scale: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Scale: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Scale tests: %v", err)
	}
}
