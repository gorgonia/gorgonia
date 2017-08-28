package tensor

import (
	"reflect"
	"testing"
	"testing/quick"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestAdd(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity)
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}
}
func TestMul(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))

		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity)
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}
}
func TestPow(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))

		correct := a.Clone().(*Dense)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity)
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}
}
func TestAdd_unsafe(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}
}
func TestMul_unsafe(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))

		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}
}
func TestPow_unsafe(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))

		correct := a.Clone().(*Dense)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}
}
func TestAdd_reuse(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
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
}
func TestMul_reuse(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
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
}
func TestPow_reuse(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
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
}
func TestAdd_incr(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}
}
func TestMul_incr(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}
}
func TestPow_incr(t *testing.T) {
	iden := func(a *Dense) bool {
		identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
		identity.Memset(identityVal(1, a.t))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden, nil); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}
}
func TestAddScalar(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity)
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(identity, a)
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity)
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(identity, a)
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestPowScalar(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity)
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_unsafe(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(identity, a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_unsafe(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(identity, a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestPowScalar_unsafe(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)

		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if ret != a {
			t.Errorf("Expected ret to be the same as a")
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_reuse(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
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
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Add(identity, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_reuse(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
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
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, numberTypes)
		ret, err := Mul(identity, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		if reuse != ret {
			t.Errorf("Expected reuse to be the same as retVal")
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestPowScalar_reuse(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
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
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_incr(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Add(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Add(identity, a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_incr(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Mul(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, numberTypes)
		ret, err := Mul(identity, a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden2, nil); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestPowScalar_incr(t *testing.T) {
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		identity := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
		copyDense(correct, a)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())

		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, identity, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Pow", a, identity, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
		}
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}

		return true
	}
	if err := quick.Check(iden1, nil); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
