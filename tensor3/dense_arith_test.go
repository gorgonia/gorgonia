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

	// basic length test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		reuse := newDense(Int, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseI) bool {
		zero := int(0)
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseI) bool {
		zero := int(0)
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseI) bool {
		zero := int(0)
		reuse := newDense(Int, x.len())
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionI(t *testing.T) {
	var f func(*QCDenseI) bool
	var err error

	// basic length test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		correct := newDense(Int, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI) bool {
		zero := newDense(Int, x.len())
		reuse := newDense(Int, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseI) bool {
		zero := int(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseI) bool {
		zero := int(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* int8 */

func TestAdditionI8(t *testing.T) {
	var f func(*QCDenseI8) bool
	var err error

	// basic length test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		reuse := newDense(Int8, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseI8) bool {
		zero := int8(0)
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseI8) bool {
		zero := int8(0)
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseI8) bool {
		zero := int8(0)
		reuse := newDense(Int8, x.len())
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionI8(t *testing.T) {
	var f func(*QCDenseI8) bool
	var err error

	// basic length test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		correct := newDense(Int8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI8) bool {
		zero := newDense(Int8, x.len())
		reuse := newDense(Int8, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseI8) bool {
		zero := int8(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseI8) bool {
		zero := int8(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* int16 */

func TestAdditionI16(t *testing.T) {
	var f func(*QCDenseI16) bool
	var err error

	// basic length test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		reuse := newDense(Int16, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseI16) bool {
		zero := int16(0)
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseI16) bool {
		zero := int16(0)
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseI16) bool {
		zero := int16(0)
		reuse := newDense(Int16, x.len())
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionI16(t *testing.T) {
	var f func(*QCDenseI16) bool
	var err error

	// basic length test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		correct := newDense(Int16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI16) bool {
		zero := newDense(Int16, x.len())
		reuse := newDense(Int16, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseI16) bool {
		zero := int16(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseI16) bool {
		zero := int16(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* int32 */

func TestAdditionI32(t *testing.T) {
	var f func(*QCDenseI32) bool
	var err error

	// basic length test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		reuse := newDense(Int32, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseI32) bool {
		zero := int32(0)
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseI32) bool {
		zero := int32(0)
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseI32) bool {
		zero := int32(0)
		reuse := newDense(Int32, x.len())
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionI32(t *testing.T) {
	var f func(*QCDenseI32) bool
	var err error

	// basic length test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		correct := newDense(Int32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI32) bool {
		zero := newDense(Int32, x.len())
		reuse := newDense(Int32, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseI32) bool {
		zero := int32(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseI32) bool {
		zero := int32(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* int64 */

func TestAdditionI64(t *testing.T) {
	var f func(*QCDenseI64) bool
	var err error

	// basic length test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		reuse := newDense(Int64, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseI64) bool {
		zero := int64(0)
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseI64) bool {
		zero := int64(0)
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseI64) bool {
		zero := int64(0)
		reuse := newDense(Int64, x.len())
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionI64(t *testing.T) {
	var f func(*QCDenseI64) bool
	var err error

	// basic length test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		correct := newDense(Int64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseI64) bool {
		zero := newDense(Int64, x.len())
		reuse := newDense(Int64, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseI64) bool {
		zero := int64(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseI64) bool {
		zero := int64(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* uint */

func TestAdditionU(t *testing.T) {
	var f func(*QCDenseU) bool
	var err error

	// basic length test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		reuse := newDense(Uint, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseU) bool {
		zero := uint(0)
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseU) bool {
		zero := uint(0)
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseU) bool {
		zero := uint(0)
		reuse := newDense(Uint, x.len())
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionU(t *testing.T) {
	var f func(*QCDenseU) bool
	var err error

	// basic length test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		correct := newDense(Uint, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU) bool {
		zero := newDense(Uint, x.len())
		reuse := newDense(Uint, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseU) bool {
		zero := uint(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseU) bool {
		zero := uint(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* uint8 */

func TestAdditionU8(t *testing.T) {
	var f func(*QCDenseU8) bool
	var err error

	// basic length test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		reuse := newDense(Uint8, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseU8) bool {
		zero := uint8(0)
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseU8) bool {
		zero := uint8(0)
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseU8) bool {
		zero := uint8(0)
		reuse := newDense(Uint8, x.len())
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionU8(t *testing.T) {
	var f func(*QCDenseU8) bool
	var err error

	// basic length test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		correct := newDense(Uint8, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU8) bool {
		zero := newDense(Uint8, x.len())
		reuse := newDense(Uint8, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseU8) bool {
		zero := uint8(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseU8) bool {
		zero := uint8(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* uint16 */

func TestAdditionU16(t *testing.T) {
	var f func(*QCDenseU16) bool
	var err error

	// basic length test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		reuse := newDense(Uint16, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseU16) bool {
		zero := uint16(0)
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseU16) bool {
		zero := uint16(0)
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseU16) bool {
		zero := uint16(0)
		reuse := newDense(Uint16, x.len())
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionU16(t *testing.T) {
	var f func(*QCDenseU16) bool
	var err error

	// basic length test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		correct := newDense(Uint16, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU16) bool {
		zero := newDense(Uint16, x.len())
		reuse := newDense(Uint16, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseU16) bool {
		zero := uint16(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseU16) bool {
		zero := uint16(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* uint32 */

func TestAdditionU32(t *testing.T) {
	var f func(*QCDenseU32) bool
	var err error

	// basic length test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		reuse := newDense(Uint32, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseU32) bool {
		zero := uint32(0)
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseU32) bool {
		zero := uint32(0)
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseU32) bool {
		zero := uint32(0)
		reuse := newDense(Uint32, x.len())
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionU32(t *testing.T) {
	var f func(*QCDenseU32) bool
	var err error

	// basic length test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		correct := newDense(Uint32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU32) bool {
		zero := newDense(Uint32, x.len())
		reuse := newDense(Uint32, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseU32) bool {
		zero := uint32(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseU32) bool {
		zero := uint32(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* uint64 */

func TestAdditionU64(t *testing.T) {
	var f func(*QCDenseU64) bool
	var err error

	// basic length test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		reuse := newDense(Uint64, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseU64) bool {
		zero := uint64(0)
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseU64) bool {
		zero := uint64(0)
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseU64) bool {
		zero := uint64(0)
		reuse := newDense(Uint64, x.len())
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionU64(t *testing.T) {
	var f func(*QCDenseU64) bool
	var err error

	// basic length test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		correct := newDense(Uint64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseU64) bool {
		zero := newDense(Uint64, x.len())
		reuse := newDense(Uint64, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseU64) bool {
		zero := uint64(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseU64) bool {
		zero := uint64(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* float32 */

func TestAdditionF32(t *testing.T) {
	var f func(*QCDenseF32) bool
	var err error

	// basic length test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		reuse := newDense(Float32, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseF32) bool {
		zero := float32(0)
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseF32) bool {
		zero := float32(0)
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseF32) bool {
		zero := float32(0)
		reuse := newDense(Float32, x.len())
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionF32(t *testing.T) {
	var f func(*QCDenseF32) bool
	var err error

	// basic length test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		correct := newDense(Float32, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseF32) bool {
		zero := newDense(Float32, x.len())
		reuse := newDense(Float32, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseF32) bool {
		zero := float32(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseF32) bool {
		zero := float32(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* float64 */

func TestAdditionF64(t *testing.T) {
	var f func(*QCDenseF64) bool
	var err error

	// basic length test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		reuse := newDense(Float64, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseF64) bool {
		zero := float64(0)
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseF64) bool {
		zero := float64(0)
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseF64) bool {
		zero := float64(0)
		reuse := newDense(Float64, x.len())
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionF64(t *testing.T) {
	var f func(*QCDenseF64) bool
	var err error

	// basic length test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		correct := newDense(Float64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseF64) bool {
		zero := newDense(Float64, x.len())
		reuse := newDense(Float64, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseF64) bool {
		zero := float64(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseF64) bool {
		zero := float64(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* complex64 */

func TestAdditionC64(t *testing.T) {
	var f func(*QCDenseC64) bool
	var err error

	// basic length test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		reuse := newDense(Complex64, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseC64) bool {
		zero := complex64(0)
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseC64) bool {
		zero := complex64(0)
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseC64) bool {
		zero := complex64(0)
		reuse := newDense(Complex64, x.len())
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionC64(t *testing.T) {
	var f func(*QCDenseC64) bool
	var err error

	// basic length test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		correct := newDense(Complex64, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseC64) bool {
		zero := newDense(Complex64, x.len())
		reuse := newDense(Complex64, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseC64) bool {
		zero := complex64(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseC64) bool {
		zero := complex64(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}

/* complex128 */

func TestAdditionC128(t *testing.T) {
	var f func(*QCDenseC128) bool
	var err error

	// basic length test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len()+1)

		_, err := zero.Add(x.Dense)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := zero.Add(x.Dense, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())

		ret, err := zero.Add(x.Dense)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		reuse := newDense(Complex128, x.len())
		ret, err := zero.Add(x.Dense, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// commutativity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)
		ret, err := x.Add(zero)
		if err != nil {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Commutativity/Associativity Identity test: %v", err)
	}

	// Safe Trans
	f = func(x *QCDenseC128) bool {
		zero := complex128(0)
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero)
		if err != nil {
			return false
		}
		if ret == x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// Unsafe Trans
	f = func(x *QCDenseC128) bool {
		zero := complex128(0)
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Trans Identity test: %v", err)
	}

	// reuse Trans
	f = func(x *QCDenseC128) bool {
		zero := complex128(0)
		reuse := newDense(Complex128, x.len())
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Trans(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}

		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Trans Identity test: %v", err)
	}
}
func TestSubtractionC128(t *testing.T) {
	var f func(*QCDenseC128) bool
	var err error

	// basic length test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len()+1)

		_, err := x.Sub(zero)
		if err == nil {
			return false
		}
		return true
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic unsafe identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if ret != x.Dense {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}

	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe Identity test: %v", err)
	}

	// basic safe identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		correct := newDense(Complex128, x.len())
		copyDense(correct, x.Dense)

		ret, err := x.Sub(zero)
		if err != nil {
			return false
		}
		if ret == zero {
			return false
		}
		return reflect.DeepEqual(ret.Data(), correct.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe Identity test: %v", err)
	}

	// reuse identity test
	f = func(x *QCDenseC128) bool {
		zero := newDense(Complex128, x.len())
		reuse := newDense(Complex128, x.len())
		ret, err := x.Sub(zero, WithReuse(reuse))
		if err != nil {
			return false
		}
		if ret != reuse {
			return false
		}
		return reflect.DeepEqual(ret.Data(), x.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Reuse Identity test: %v", err)
	}

	// TransInv - tests of commutativity  - safe one
	f = func(x *QCDenseC128) bool {
		zero := complex128(0)
		xminus, err := x.TransInv(zero)
		if err != nil {
			return false
		}

		zerominus, err := x.TransInvR(zero)
		if err != nil {
			return false
		}
		return !reflect.DeepEqual(xminus.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Safe TransInc test: %v", err)
	}

	// TransInv - tests of commutativity  - unsafe
	f = func(x *QCDenseC128) bool {
		zero := complex128(0)
		cloned := x.Clone().(*Dense)
		xminus, err := x.TransInv(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if xminus != x.Dense {
			return false
		}

		zerominus, err := x.TransInvR(zero, UseUnsafe())
		if err != nil {
			return false
		}
		if zerominus != x.Dense {
			return false
		}
		return !reflect.DeepEqual(cloned.Data(), zerominus.Data())
	}
	if err = quick.Check(f, nil); err != nil {
		t.Errorf("Failed Unsafe TransInv test: %v", err)
	}
}
