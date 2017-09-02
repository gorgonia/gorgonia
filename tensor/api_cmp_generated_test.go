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

func TestGt(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Gt(a, b)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gt(b, c)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gt(a, c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gt failed: %v", err)
	}

}
func TestGte(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Gte(a, b)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gte(b, c)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gte(a, c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gte failed: %v", err)
	}

}
func TestLt(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Lt(a, b)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lt(b, c)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lt(a, c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lt failed: %v", err)
	}

}
func TestLte(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Lte(a, b)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lte(b, c)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lte(a, c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lte failed: %v", err)
	}

}
func TestEq(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := ElEq(a, b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := ElEq(b, c)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := ElEq(a, c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := ElEq(a, b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElEq(b, a)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}
}
func TestNe(t *testing.T) {
	var r *rand.Rand
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := ElNe(a, b)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElNe(b, a)
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElNe failed: %v", err)
	}
}
func TestGt_assame(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Gt(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gt(b, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gt(a, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gt failed: %v", err)
	}

}
func TestGte_assame(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Gte(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gte(b, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gte(a, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gte failed: %v", err)
	}

}
func TestLt_assame(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Lt(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lt(b, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lt(a, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lt failed: %v", err)
	}

}
func TestLte_assame(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := Lte(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lte(b, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lte(a, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lte failed: %v", err)
	}

}
func TestEq_assame(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := ElEq(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := ElEq(b, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := ElEq(a, c, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := ElEq(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElEq(b, a, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}
}
func TestNe_assame(t *testing.T) {
	var r *rand.Rand
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := ElNe(a, b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElNe(b, a, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElNe failed: %v", err)
	}
}
func TestGtScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := Gt(a, b)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gt(b, c)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gt(a, c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gt failed: %v", err)
	}

}
func TestGteScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := Gte(a, b)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Gte(b, c)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Gte(a, c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Gte failed: %v", err)
	}

}
func TestLtScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := Lt(a, b)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lt(b, c)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lt(a, c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lt failed: %v", err)
	}

}
func TestLteScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := Lte(a, b)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := Lte(b, c)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := Lte(a, c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for Lte failed: %v", err)
	}

}
func TestEqScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := ElEq(a, b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := ElEq(b, c)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := ElEq(a, c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.(*Dense).Bools()
		bc := bxc.(*Dense).Bools()
		ac := axc.(*Dense).Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := ElEq(a, b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElEq(b, a)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElEq failed: %v", err)
	}
}
func TestNeScalar(t *testing.T) {
	var r *rand.Rand
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := ElNe(a, b)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := ElNe(b, a)
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	r = rand.New(rand.NewSource(time.Now().UnixNano()))
	if err := quick.Check(symFn, &quick.Config{Rand: r}); err != nil {
		t.Error("Transitivity test for ElNe failed: %v", err)
	}
}
