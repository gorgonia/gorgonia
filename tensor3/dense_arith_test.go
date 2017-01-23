package tensor

import (
	"reflect"
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

/* int */

func TestAdditionI(t *testing.T) {
	var f func(*QCDenseI) bool
	var err error

	f = func(x *QCDenseI) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int, x.len())
		correct = newDense(Int, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Int, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI) bool {
		var ret, correct, reuse *Dense
		var zero int
		var err error
		zero = 0
		correct = newDense(Int, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Int, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionI8(t *testing.T) {
	var f func(*QCDenseI8) bool
	var err error

	f = func(x *QCDenseI8) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int8, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int8, x.len())
		correct = newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int8, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int8, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Int8, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI8) bool {
		var ret, correct, reuse *Dense
		var zero int8
		var err error
		zero = 0
		correct = newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Int8, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionI16(t *testing.T) {
	var f func(*QCDenseI16) bool
	var err error

	f = func(x *QCDenseI16) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int16, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int16, x.len())
		correct = newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int16, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int16, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Int16, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI16) bool {
		var ret, correct, reuse *Dense
		var zero int16
		var err error
		zero = 0
		correct = newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Int16, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionI32(t *testing.T) {
	var f func(*QCDenseI32) bool
	var err error

	f = func(x *QCDenseI32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int32, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int32, x.len())
		correct = newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int32, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int32, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Int32, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI32) bool {
		var ret, correct, reuse *Dense
		var zero int32
		var err error
		zero = 0
		correct = newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Int32, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionI64(t *testing.T) {
	var f func(*QCDenseI64) bool
	var err error

	f = func(x *QCDenseI64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Int64, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Int64, x.len())
		correct = newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Int64, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Int64, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Int64, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseI64) bool {
		var ret, correct, reuse *Dense
		var zero int64
		var err error
		zero = 0
		correct = newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Int64, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionU(t *testing.T) {
	var f func(*QCDenseU) bool
	var err error

	f = func(x *QCDenseU) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint, x.len())
		correct = newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Uint, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU) bool {
		var ret, correct, reuse *Dense
		var zero uint
		var err error
		zero = 0
		correct = newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Uint, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionU8(t *testing.T) {
	var f func(*QCDenseU8) bool
	var err error

	f = func(x *QCDenseU8) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint8, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint8, x.len())
		correct = newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint8, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint8, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Uint8, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU8) bool {
		var ret, correct, reuse *Dense
		var zero uint8
		var err error
		zero = 0
		correct = newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Uint8, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionU16(t *testing.T) {
	var f func(*QCDenseU16) bool
	var err error

	f = func(x *QCDenseU16) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint16, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint16, x.len())
		correct = newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint16, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint16, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Uint16, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU16) bool {
		var ret, correct, reuse *Dense
		var zero uint16
		var err error
		zero = 0
		correct = newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Uint16, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionU32(t *testing.T) {
	var f func(*QCDenseU32) bool
	var err error

	f = func(x *QCDenseU32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint32, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint32, x.len())
		correct = newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint32, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint32, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Uint32, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU32) bool {
		var ret, correct, reuse *Dense
		var zero uint32
		var err error
		zero = 0
		correct = newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Uint32, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionU64(t *testing.T) {
	var f func(*QCDenseU64) bool
	var err error

	f = func(x *QCDenseU64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Uint64, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Uint64, x.len())
		correct = newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Uint64, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Uint64, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Uint64, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseU64) bool {
		var ret, correct, reuse *Dense
		var zero uint64
		var err error
		zero = 0
		correct = newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Uint64, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionF32(t *testing.T) {
	var f func(*QCDenseF32) bool
	var err error

	f = func(x *QCDenseF32) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Float32, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Float32, x.len())
		correct = newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float32, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Float32, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Float32, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseF32) bool {
		var ret, correct, reuse *Dense
		var zero float32
		var err error
		zero = 0
		correct = newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Float32, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionF64(t *testing.T) {
	var f func(*QCDenseF64) bool
	var err error

	f = func(x *QCDenseF64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Float64, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Float64, x.len())
		correct = newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Float64, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Float64, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Float64, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseF64) bool {
		var ret, correct, reuse *Dense
		var zero float64
		var err error
		zero = 0
		correct = newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Float64, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionC64(t *testing.T) {
	var f func(*QCDenseC64) bool
	var err error

	f = func(x *QCDenseC64) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Complex64, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Complex64, x.len())
		correct = newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex64, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Complex64, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Complex64, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseC64) bool {
		var ret, correct, reuse *Dense
		var zero complex64
		var err error
		zero = 0
		correct = newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Complex64, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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

func TestAdditionC128(t *testing.T) {
	var f func(*QCDenseC128) bool
	var err error

	f = func(x *QCDenseC128) bool {
		var ret, correct, zero, reuse *Dense
		var err error
		zero = newDense(Complex128, x.len()+1)

		// basic length test
		if _, err = zero.Add(x.Dense); err == nil {
			t.Error("Failed length test")
			return false
		}

		// safe identity
		zero = newDense(Complex128, x.len())
		correct = newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		if ret, err = zero.Add(x.Dense); err != nil {
			t.Errorf("Failed safe identity test: %v", err)
			return false
		}

		if ret == zero || ret == x.Dense {
			t.Error("Failed safe identity test: safe op should not return same value")
			return false
		}

		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed safe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// reuse identity - wrong length
		reuse = newDense(Complex128, x.len()+1)
		if _, err = x.Add(zero, WithReuse(reuse)); err == nil {
			t.Error("Expected an error when reuse is of a differing size")
			return false
		}

		// reuse identity
		reuse = newDense(Complex128, x.len())
		if ret, err = zero.Add(x.Dense, WithReuse(reuse)); err != nil {
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
		if ret, err = zero.Add(x.Dense, UseUnsafe()); err != nil {
			t.Errorf("Failed unsafe identity test: %v", err)
			return false
		}
		if ret != zero {
			t.Error("Failed unsafe identity test. Expected the return *Dense to be the same as the left")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		// test commutativity and associativity
		zero = newDense(Complex128, x.len())
		if ret, err = x.Add(zero); err != nil {
			t.Errorf("Failed commutativity/associativity test: %v", err)
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed unsafe identity test: operation incorrect: \nWant: %v \nGot: %v\n", correct.Data(), ret.Data())
			return false
		}

		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	f = func(x *QCDenseC128) bool {
		var ret, correct, reuse *Dense
		var zero complex128
		var err error
		zero = 0
		correct = newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		// Safe Trans
		if ret, err = x.Trans(zero); err != nil {
			t.Errorf("Failed Safe Trans test %v", err)
			return false
		}
		if ret == x.Dense {
			t.Error("Failed Safe Trans: ret == x")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Safe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Reuse Trans
		reuse = newDense(Complex128, x.len())
		if ret, err = x.Trans(zero, WithReuse(reuse)); err != nil {
			t.Errorf("Failed Reuse Trans test %v", err)
			return false
		}
		if ret != reuse {
			t.Error("Failed Reuse Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Reuse Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}

		// Unsafe Trans
		if ret, err = x.Trans(zero, UseUnsafe()); err != nil {
			t.Errorf("Unsafe trans test failed:  %v", err)
			return false
		}
		if ret != x.Dense {
			t.Error("Failed Unsafe Trans: expected return value to be the same as reuse")
			return false
		}
		if !reflect.DeepEqual(correct.Data(), ret.Data()) {
			t.Errorf("Failed Unsafe Trans: Operation incorrect. \nWant %v\nGot%v", correct.Data(), ret.Data())
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Trans tests: %v", err)
	}
}
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
