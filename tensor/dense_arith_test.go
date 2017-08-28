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

func TestDense_Add(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Add(b)
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
func TestDense_Sub(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Sub(b)
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Add(b, UseUnsafe())

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
func TestDense_Mul(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Mul(b)
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
func TestDense_Div(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Div(b)
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Mul(b, UseUnsafe())

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
func TestDense_Pow(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.Pow(b)
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
func TestDense_Add_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Add(b, UseUnsafe())
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
func TestDense_Sub_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Sub(b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Add(b, UseUnsafe())

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
func TestDense_Mul_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Mul(b, UseUnsafe())
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
func TestDense_Div_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Div(b, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Mul(b, UseUnsafe())

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
func TestDense_Pow_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.Pow(b, UseUnsafe())
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
func TestDense_Add_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Add(b, WithReuse(reuse))
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
func TestDense_Sub_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Sub(b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Add(b, UseUnsafe())

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
func TestDense_Mul_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Mul(b, WithReuse(reuse))
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
func TestDense_Div_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.Div(b, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Mul(b, UseUnsafe())

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
func TestDense_Pow_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		b.Memset(identityVal(1, a.t))
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.Pow(b, WithReuse(reuse))
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
func TestDense_Add_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := a.Add(b, WithIncr(incr))
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
func TestDense_Sub_incr(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv := func(a *Dense) bool {
		b := New(Of(a.t), WithShape(a.Shape().Clone()...))
		incr := New(Of(a.t), WithShape(a.Shape().Clone()...))
		correct := a.Clone().(*Dense)
		incr.Memset(identityVal(100, a.t))
		correct.Add(incr, UseUnsafe())
		we := willerr(a, numberTypes)
		ret, err := a.Sub(b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Sub", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Add(b, UseUnsafe())

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
func TestDense_Mul_incr(t *testing.T) {
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
		ret, err := a.Mul(b, WithIncr(incr))
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
func TestDense_Div_incr(t *testing.T) {
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
		ret, err := a.Div(b, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "Div", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.Mul(b, UseUnsafe())

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
func TestDense_Pow_incr(t *testing.T) {
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
		ret, err := a.Pow(b, WithIncr(incr))
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
func TestDense_AddScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.AddScalar(b, true)
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
		ret, err := a.AddScalar(b, false)
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
func TestDense_SubScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.SubScalar(b, true)
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.AddScalar(b, true, UseUnsafe())

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
		ret, err := a.SubScalar(b, false)
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.SubScalar(b, false, UseUnsafe())

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
func TestDense_MulScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.MulScalar(b, true)
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
		ret, err := a.MulScalar(b, false)
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
func TestDense_DivScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.DivScalar(b, true)
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.MulScalar(b, true, UseUnsafe())

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
func TestDense_PowScalar(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.PowScalar(b, true)
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
func TestDense_AddScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.AddScalar(b, true, UseUnsafe())
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
		ret, err := a.AddScalar(b, false, UseUnsafe())
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
func TestDense_SubScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.SubScalar(b, true, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.AddScalar(b, true, UseUnsafe())

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
		ret, err := a.SubScalar(b, false, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.SubScalar(b, false, UseUnsafe())

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
func TestDense_MulScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.MulScalar(b, true, UseUnsafe())
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
		ret, err := a.MulScalar(b, false, UseUnsafe())
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
func TestDense_DivScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.DivScalar(b, true, UseUnsafe())
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.MulScalar(b, true, UseUnsafe())

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
func TestDense_PowScalar_unsafe(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.PowScalar(b, true, UseUnsafe())
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
func TestDense_AddScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.AddScalar(b, true, WithReuse(reuse))
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
		ret, err := a.AddScalar(b, false, WithReuse(reuse))
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
func TestDense_SubScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(0, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.SubScalar(b, true, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.AddScalar(b, true, UseUnsafe())

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
		ret, err := a.SubScalar(b, false, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.SubScalar(b, false, UseUnsafe())

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
func TestDense_MulScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.MulScalar(b, true, WithReuse(reuse))
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
		ret, err := a.MulScalar(b, false, WithReuse(reuse))
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
func TestDense_DivScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	inv1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, numberTypes)
		ret, err := a.DivScalar(b, true, WithReuse(reuse))
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.MulScalar(b, true, UseUnsafe())

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
func TestDense_PowScalar_reuse(t *testing.T) {
	var r *rand.Rand
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	iden1 := func(q *Dense) bool {
		a := q.Clone().(*Dense)
		b := identityVal(1, q.t)
		reuse := New(Of(a.t), WithShape(a.Shape().Clone()...))

		correct := a.Clone().(*Dense)
		we := willerr(a, floatcmplxTypes)
		ret, err := a.PowScalar(b, true, WithReuse(reuse))
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
func TestDense_AddScalar_incr(t *testing.T) {
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
		ret, err := a.AddScalar(b, true, WithIncr(incr))
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
		ret, err := a.AddScalar(b, false, WithIncr(incr))
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
func TestDense_SubScalar_incr(t *testing.T) {
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
		ret, err := a.SubScalar(b, true, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "SubVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.AddScalar(b, true, UseUnsafe())

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
		ret, err := a.SubScalar(b, false, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "SubSV", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.SubScalar(b, false, UseUnsafe())

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
func TestDense_MulScalar_incr(t *testing.T) {
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
		ret, err := a.MulScalar(b, true, WithIncr(incr))
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
		ret, err := a.MulScalar(b, false, WithIncr(incr))
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
func TestDense_DivScalar_incr(t *testing.T) {
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
		ret, err := a.DivScalar(b, true, WithIncr(incr))
		if err, retEarly := qcErrCheck(t, "DivVS", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		ret, err = ret.MulScalar(b, true, UseUnsafe())

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
func TestDense_PowScalar_incr(t *testing.T) {
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
		ret, err := a.PowScalar(b, true, WithIncr(incr))
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
