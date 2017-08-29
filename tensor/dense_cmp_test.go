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

func TestDense_Gt(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gt(b)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_Gte(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gte(b)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_Lt(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lt(b)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_Lte(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lte(b)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_Eq(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.ElEq(b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElEq(b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElEq(a)
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
func TestDense_Ne(t *testing.T) {
	var r *rand.Rand
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElNe(b)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElNe(a)
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
func TestDense_GtScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GtScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GtScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_GteScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GteScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GteScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_LtScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LtScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LtScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_LteScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LteScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LteScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
func TestDense_EqScalar(t *testing.T) {
	var r *rand.Rand
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.ElEqScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.ElEqScalar(b, false)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
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
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElEqScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElEqScalar(b, false)
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
func TestDense_NeScalar(t *testing.T) {
	var r *rand.Rand
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElNeScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElNeScalar(b, false)
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
