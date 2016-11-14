package gorgonia

import (
	"fmt"
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
)

func TestDtypeBasics(t *testing.T) {
	assert := assert.New(t)

	var t0 Dtype
	var a hm.TypeVariable

	t0 = Float64
	a = hm.NewTypeVar("a")

	assert.False(t0.Contains(a))
	assert.True(t0.Eq(Float64))
	assert.False(t0.Eq(Float32))
	assert.False(t0.Eq(a))
	assert.Nil(t0.Types())
	assert.Equal(Float64, t0.Clone())
	assert.Equal(Float64, t0.Replace(a, Float32))
	assert.True(t0.IsConstant())

	// for completeness sake
	assert.Equal("Float64", t0.String())
	assert.Equal("Float64", fmt.Sprintf("%v", t0))
}

func TestDtypeOps(t *testing.T) {
	assert := assert.New(t)

	var t0, t1 hm.Type
	var a hm.TypeVariable
	var err error

	a = hm.NewTypeVar("a")

	if t0, t1, _, err = hm.Unify(a, Float64); err != nil {
		t.Fatal(err)
	}

	assert.Equal(Float64, hm.Prune(t0))
	assert.Equal(Float64, hm.Prune(t1))

	if t0, t1, _, err = hm.Unify(Float64, a); err != nil {
		t.Fatal(err)
	}
	assert.Equal(Float64, hm.Prune(t0))
	assert.Equal(Float64, hm.Prune(t1))
}

var tensorTypeTests []struct {
	a, b TensorType

	eq        bool
	containsA bool
	containsB bool
	types     hm.Types
	replaced  TensorType
}

func TestTensorTypeBasics(t *testing.T) {
	assert := assert.New(t)

	for _, ttts := range tensorTypeTests {
		// Equality
		if ttts.eq {
			assert.True(ttts.a.Eq(ttts.b))
		} else {
			assert.False(ttts.a.Eq(ttts.b))
		}

		// Contains
		tv := hm.NewTypeVar("a")
		if ttts.containsA {
			assert.True(ttts.a.Contains(tv))
		} else {
			assert.False(ttts.a.Contains(tv))
		}

		// Contains
		tv = hm.NewTypeVar("b")
		if ttts.containsB {
			assert.True(ttts.a.Contains(tv))
		} else {
			assert.False(ttts.a.Contains(tv))
		}

		// Types
		assert.Equal(ttts.types, ttts.a.Types())

		// Clone
		assert.Equal(ttts.a, ttts.a.Clone())

		// Replace
		tv = hm.NewTypeVar("a")
		assert.Equal(ttts.replaced, ttts.a.Replace(tv, Float64))

		// string and format for completeness sake
		assert.Equal("Tensor", ttts.a.Name())
		if ttts.containsA {
			assert.Equal("Tensor a", ttts.a.String())
		} else {
			assert.Equal("Tensor Float64", ttts.a.String())
		}
	}

}

var tensorOpsTest []struct {
	name string

	a hm.Type
	b hm.Type

	aPrime hm.Type
	bPrime hm.Type
}

func TestTensorTypeOps(t *testing.T) {
	assert := assert.New(t)

	for _, tots := range tensorOpsTest {
		ap, bp, _, err := hm.Unify(tots.a, tots.b)
		if err != nil {
			t.Error(err)
			continue
		}

		assert.True(tots.aPrime.Eq(hm.Prune(ap)), "Test %q: Wanted: %#v. Got %#v", tots.name, tots.aPrime, ap)
		assert.True(tots.bPrime.Eq(hm.Prune(bp)), "Test %q: Wanted: %#v. Got %#v", tots.name, tots.bPrime, bp)
	}

}

func init() {
	tensorTypeTests = []struct {
		a, b TensorType

		eq        bool
		containsA bool
		containsB bool
		types     hm.Types
		replaced  TensorType
	}{

		{newTensorType(1, Float64), newTensorType(1, Float64), true, false, false, hm.Types{Float64}, newTensorType(1, Float64)},
		{newTensorType(1, Float64), newTensorType(1, Float32), false, false, false, hm.Types{Float64}, newTensorType(1, Float64)},
		{newTensorType(1, Float64), newTensorType(2, Float64), false, false, false, hm.Types{Float64}, newTensorType(1, Float64)},
		{newTensorType(1, hm.NewTypeVar("a")), newTensorType(1, hm.NewTypeVar("a")), true, true, false, hm.Types{hm.NewTypeVar("a")}, newTensorType(1, Float64)},
		{newTensorType(1, hm.NewTypeVar("a")), newTensorType(1, hm.NewTypeVar("b")), false, true, false, hm.Types{hm.NewTypeVar("a")}, newTensorType(1, Float64)},
	}

	tensorOpsTest = []struct {
		name string

		a hm.Type
		b hm.Type

		aPrime hm.Type
		bPrime hm.Type
	}{
		{"a ⋃ Tensor Float64", hm.NewTypeVar("a"), newTensorType(1, Float64), newTensorType(1, Float64), newTensorType(1, Float64)},
		{"Tensor Float64 ⋃ a", newTensorType(1, Float64), hm.NewTypeVar("a"), newTensorType(1, Float64), newTensorType(1, Float64)},
		{"Tensor a ⋃ Tensor Float64", newTensorType(1, hm.NewTypeVar("a")), newTensorType(1, Float64), newTensorType(1, Float64), newTensorType(1, Float64)},
		{"Tensor a ⋃ Tensor Float64", newTensorType(1, Float64), newTensorType(1, hm.NewTypeVar("a")), newTensorType(1, Float64), newTensorType(1, Float64)},
	}
}
