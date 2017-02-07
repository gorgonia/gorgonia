package gorgonia

import (
	"fmt"
	"testing"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
)

func TestDtypeBasics(t *testing.T) {
	assert := assert.New(t)

	var t0 tensor.Dtype
	var a hm.TypeVariable

	t0 = Float64
	a = hm.TypeVariable('a')

	assert.True(t0.Eq(Float64))
	assert.False(t0.Eq(Float32))
	assert.False(t0.Eq(a))
	assert.Nil(t0.Types())

	k := hm.TypeVarSet{'x', 'y'}
	v := hm.TypeVarSet{'a', 'b'}
	t1, err := t0.Normalize(k, v)
	assert.Nil(err)
	assert.Equal(t0, t1)

	// for completeness sake
	assert.Equal("float64", t0.Name())
	assert.Equal("float64", t0.String())
	assert.Equal("float64", fmt.Sprintf("%v", t0))

}

func TestDtypeOps(t *testing.T) {
	var sub hm.Subs
	var a hm.TypeVariable
	var err error

	a = hm.TypeVariable('a')

	if sub, err = hm.Unify(a, Float64); err != nil {
		t.Fatal(err)
	}

	if repl, ok := sub.Get(a); !ok {
		t.Errorf("Expected a substitution for %v", a)
	} else if repl != Float64 {
		t.Errorf("Expecetd substitution for %v to be %v. Got %v instead", a, Float64, repl)
	}

	if sub, err = hm.Unify(Float64, a); err != nil {
		t.Fatal(err)
	}

	if repl, ok := sub.Get(a); !ok {
		t.Errorf("Expected a substitution for %v", a)
	} else if repl != Float64 {
		t.Errorf("Expecetd substitution for %v to be %v. Got %v instead", a, Float64, repl)
	}
}

var tensorTypeTests []struct {
	a, b TensorType

	eq     bool
	types  hm.Types
	format string
}

func TestTensorTypeBasics(t *testing.T) {
	assert := assert.New(t)

	for _, ttts := range tensorTypeTests {
		// Equality
		if ttts.eq {
			assert.True(ttts.a.Eq(ttts.b), "TensorType Equality failed: %#v != %#v", ttts.a, ttts.b)
		} else {
			assert.False(ttts.a.Eq(ttts.b), "TensorType Equality: %v == %v should be false", ttts.a, ttts.b)
		}

		// Types
		assert.Equal(ttts.types, ttts.a.Types())

		// string and format for completeness sake
		assert.Equal("Tensor", ttts.a.Name())
		assert.Equal(ttts.format, fmt.Sprintf("%v", ttts.a))
		assert.Equal(fmt.Sprintf("Tensor-%d %v", ttts.a.Dims, ttts.a.Of), fmt.Sprintf("%#v", ttts.a))
	}

	tt := newTensorType(1, hm.TypeVariable('x'))
	k := hm.TypeVarSet{'x', 'y'}
	v := hm.TypeVarSet{'a', 'b'}
	tt2, err := tt.Normalize(k, v)
	if err != nil {
		t.Error(err)
	}
	assert.True(tt2.Eq(newTensorType(1, hm.TypeVariable('a'))))

}

var tensorOpsTest []struct {
	name string

	a hm.Type
	b hm.Type

	aSub hm.Type
}

func TestTensorTypeOps(t *testing.T) {
	for _, tots := range tensorOpsTest {
		sub, err := hm.Unify(tots.a, tots.b)
		if err != nil {
			t.Error(err)
			continue
		}

		if subst, ok := sub.Get(hm.TypeVariable('a')); !ok {
			t.Errorf("Expected a substitution for a")
		} else if !subst.Eq(tots.aSub) {
			t.Errorf("Expected substitution to be %v. Got %v instead", tots.aSub, subst)
		}
	}
}

func init() {
	tensorTypeTests = []struct {
		a, b TensorType

		eq     bool
		types  hm.Types
		format string
	}{

		{newTensorType(1, Float64), newTensorType(1, Float64), true, hm.Types{Float64}, "Vector float64"},
		{newTensorType(1, Float64), newTensorType(1, Float32), false, hm.Types{Float64}, "Vector float64"},
		{newTensorType(1, Float64), newTensorType(2, Float64), false, hm.Types{Float64}, "Vector float64"},
		{newTensorType(1, hm.TypeVariable('a')), newTensorType(1, hm.TypeVariable('a')), true, hm.Types{hm.TypeVariable('a')}, "Vector a"},
		{newTensorType(1, hm.TypeVariable('a')), newTensorType(1, hm.TypeVariable('b')), false, hm.Types{hm.TypeVariable('a')}, "Vector a"},
	}

	tensorOpsTest = []struct {
		name string

		a hm.Type
		b hm.Type

		aSub hm.Type
	}{
		{"a ~ Tensor Float64", hm.TypeVariable('a'), newTensorType(1, Float64), newTensorType(1, Float64)},
		{"Tensor Float64 ~ a", newTensorType(1, Float64), hm.TypeVariable('a'), newTensorType(1, Float64)},
		{"Tensor a ~ Tensor Float64", newTensorType(1, hm.TypeVariable('a')), newTensorType(1, Float64), Float64},
		{"Tensor a ~ Tensor Float64", newTensorType(1, Float64), newTensorType(1, hm.TypeVariable('a')), Float64},
	}
}
