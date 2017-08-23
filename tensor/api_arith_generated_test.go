package tensor

import (
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestAdd(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Add(a, identity)
		if err != nil {
			t.Errorf("Identity tests for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Add(a, identity)
		if err != nil {
			t.Errorf("Identity sliced test for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Add failed: %v", err)
	}
}
func TestMul(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Mul(a, identity)
		if err != nil {
			t.Errorf("Identity tests for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Mul(a, identity)
		if err != nil {
			t.Errorf("Identity sliced test for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Mul failed: %v", err)
	}
}
func TestPow(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Pow(a, identity)
		if err != nil {
			t.Errorf("Identity tests for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Pow(a, identity)
		if err != nil {
			t.Errorf("Identity sliced test for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Pow failed: %v", err)
	}
}
func TestAdd_unsafe(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Add(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity tests for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Add(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity sliced test for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Add failed: %v", err)
	}
}
func TestMul_unsafe(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Mul(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity tests for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Mul(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity sliced test for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Mul failed: %v", err)
	}
}
func TestPow_unsafe(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Pow(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity tests for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)

		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Pow(a, identity, UseUnsafe())
		if err != nil {
			t.Errorf("Identity sliced test for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a.Dense")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Pow failed: %v", err)
	}
}
func TestAdd_reuse(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Add(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity tests for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Add(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity sliced test for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Add failed: %v", err)
	}
}
func TestMul_reuse(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Mul(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity tests for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Mul(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity sliced test for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Mul failed: %v", err)
	}
}
func TestPow_reuse(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)

		ret, err := Pow(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity tests for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		reuse := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)

		ret, err := Pow(a, identity, WithReuse(reuse))
		if err != nil {
			t.Errorf("Identity sliced test for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Pow failed: %v", err)
	}
}
func TestAdd_incr(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Add(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity tests for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Add(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity sliced test for Add was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Add failed: %v", err)
	}
}
func TestMul_incr(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Mul(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity tests for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Mul(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity sliced test for Mul was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Mul failed: %v", err)
	}
}
func TestPow_incr(t *testing.T) {
	iden := func(a *QCDenseF64) bool {
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(a.len()))
		copyDense(correct, a)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Pow(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity tests for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0, 5))
		identity := New(Of(Float64), WithShape(a.len()))
		identity.Memset(1.0)
		incr := New(Of(Float64), WithShape(a.len()))
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)
		incr.Memset(100.0)
		dat := correct.Data().([]float64)
		for i := range dat {
			dat[i] += 100.0
		}

		ret, err := Pow(a, identity, WithIncr(incr))
		if err != nil {
			t.Errorf("Identity sliced test for Pow was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil {
		t.Errorf("IdentitySliced test for Pow failed: %v", err)
	}
}
