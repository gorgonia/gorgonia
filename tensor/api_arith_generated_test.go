package tensor

import (
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
	"time"
)

/*
GENERATED FILE. DO NOT EDIT
*/

func TestAdd(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b)
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

}
func TestSub(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b)
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub failed: %v", err)
	}
}
func TestMul(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b)
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

}
func TestDiv(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b)
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div failed: %v", err)
	}
}
func TestPow(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b)
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

}
func TestAdd_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

}
func TestSub_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub failed: %v", err)
	}
}
func TestMul_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

}
func TestDiv_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div failed: %v", err)
	}
}
func TestPow_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

}
func TestAdd_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

}
func TestSub_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub failed: %v", err)
	}
}
func TestMul_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

}
func TestDiv_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div failed: %v", err)
	}
}
func TestPow_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

}
func TestAdd_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add failed: %v", err)
	}

}
func TestSub_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub failed: %v", err)
	}
}
func TestMul_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul failed: %v", err)
	}

}
func TestDiv_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div failed: %v", err)
	}
}
func TestPow_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("a.Dtype: %v", a.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow failed: %v", err)
	}

}
func TestAddScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b)
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(b, a)
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestSubScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b)
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (tensor as left, scalar as right) failed: %v", err)
	}

	inv2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(b, a)
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Sub(b, ret, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(inv2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b)
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(b, a)
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestDivScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b)
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestPowScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b)
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(b, a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestSubScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (tensor as left, scalar as right) failed: %v", err)
	}

	inv2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(b, a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Sub(b, ret, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(inv2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(b, a, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestDivScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestPowScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Add(b, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestSubScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (tensor as left, scalar as right) failed: %v", err)
	}

	inv2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Sub(b, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Sub(b, ret, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(inv2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Mul(b, a, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestDivScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
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
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestPowScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
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

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestAddScalar_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Add(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Add(b, a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Add", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Add (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestSubScalar_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Sub(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Add(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (tensor as left, scalar as right) failed: %v", err)
	}

	inv2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Sub(b, a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Sub(b, ret, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(inv2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Sub (scalar as left, tensor as right) failed: %v", err)
	}
}
func TestMulScalar_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Mul(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (tensor as left, scalar as right) failed: %v", err)
	}

	iden2 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Mul(b, a, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Mul", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(iden2, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Mul (scalar as left, tensor as right) failed: %v", err)
	}

}
func TestDivScalar_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := Div(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = Mul(ret, b, UseUnsafe())

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}
	if err := quick.Check(inv1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Inv test for Div (tensor as left, scalar as right) failed: %v", err)
	}

}
func TestPowScalar_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, floatcmplxTypes)
		ret, err := Pow(a, b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Pow", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		isFloatTypes := qcIsFloat(a)
		if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
			t.Errorf("q.Dtype: %v", q.Dtype())
			t.Errorf("Correct.Data()\n%v", correct.Data())
			t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		return true
	}

	if err := quick.Check(iden1, &quick.Config{Rand: r}); err != nil {
		t.Errorf("Identity test for Pow (tensor as left, scalar as right) failed: %v", err)
	}

}
