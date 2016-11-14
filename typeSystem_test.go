package gorgonia

import (
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
)

var scalarTypeTests []struct {
	name string
	a    hm.Type

	isScalar bool
	panics   bool
}

func TestIsScalarType(t *testing.T) {
	for _, stts := range scalarTypeTests {
		if stts.panics {
			f := func() {
				isScalarType(stts.a)
			}
			assert.Panics(t, f)
			continue
		}

		if isScalarType(stts.a) != stts.isScalar {
			t.Errorf("Expected isScalarType(%v) to be scalar: %v", stts.a, stts.isScalar)
		}
	}
}

var dtypeOfTests []struct {
	a hm.Type

	correct Dtype
	err     bool
}

func TestDtypeOf(t *testing.T) {
	for _, dots := range dtypeOfTests {
		dt, err := dtypeOf(dots.a)

		switch {
		case err != nil && !dots.err:
			t.Errorf("Error when performing dtypeOf(%v): %v", dots.a, err)
		case err == nil && dots.err:
			t.Errorf("Expected an error when performing dtypeOf(%v)", dots.a)
		}

		if dots.err {
			continue
		}

		if !dots.correct.Eq(dt) {
			t.Errorf("Incorrect dtypeOf when performing dtypeOf(%v). Expected %v. Got %v", dots.a, dots.correct, dt)
		}

	}
}

func init() {
	scalarTypeTests = []struct {
		name string
		a    hm.Type

		isScalar bool
		panics   bool
	}{
		{"a:Float64", hm.NewTypeVar("a", hm.WithInstance(Float64)), true, false},
		{"Float64", Float64, true, false},
		{"Tensor Float64", newTensorType(1, Float64), false, false},
		{"Tensor Float64 (special)", newTensorType(0, Float64), true, false},

		// bad shit
		{"a", hm.NewTypeVar("a"), false, true},
		{"malformed", malformed{}, false, true},
	}

	dtypeOfTests = []struct {
		a hm.Type

		correct Dtype
		err     bool
	}{
		// {Float64, Float64, false},
		// {newTensorType(1, Float64), Float64, false},
		// {hm.NewTypeVar("a", hm.WithInstance(Float64)), Float64, false},
		// {hm.NewTypeVar("a", hm.WithInstance(newTensorType(1, Float64))), Float64, false},

		// bad shit
		{hm.NewTypeVar("a"), MAXDTYPE, true},
		{newTensorType(1, hm.NewTypeVar("a")), MAXDTYPE, true},
	}
}
