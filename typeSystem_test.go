package gorgonia

import (
	"testing"

	"github.com/chewxy/hm"
	"github.com/stretchr/testify/assert"
)

// TODO: gather edge cases
func TestInferNodeType(t *testing.T) {
	assert := assert.New(t)
	g := NewGraph()
	var inferNodeTests = []struct {
		name     string
		op       Op
		children Nodes

		correct hm.Type
		err     bool
	}{
		// simple case Float+Float
		{"+(1, 2)",
			newEBOByType(addOpType, Float64, Float64),
			Nodes{
				newNode(withGraph(g), withType(Float64), WithName("a")),
				newNode(withGraph(g), withType(Float64), WithName("b"))},
			Float64,
			false},

		// complicated case: will error out due to mis match
		{"+(1, 2)",
			newEBOByType(addOpType, Float64, Float32),
			Nodes{
				newNode(withGraph(g), withType(Float64), WithName("a")),
				newNode(withGraph(g), withType(Float32), WithName("b"))},
			Float64,
			true},
	}

	for _, ints := range inferNodeTests {
		t0, err := inferNodeType(ints.op, ints.children...)
		switch {
		case ints.err && err == nil:
			t.Errorf("Expected an error in test %q", ints.name)
		case !ints.err && err != nil:
			t.Errorf("Error in test %q: %v", ints.name, err)
		}

		if ints.err {
			continue
		}

		assert.True(ints.correct.Eq(t0))
	}
}

var inferTypeTests = []struct {
	expr interface{}

	correct hm.Type
	err     bool
}{
	{newEBOByType(addOpType, Float64, Float64), hm.NewFnType(hm.NewTypeVar("a"), hm.NewTypeVar("a"), hm.NewTypeVar("a")), false},
	{float32(0), Float32, false},
	{float64(0), Float64, false},
	{0, Int, false},
	{int64(0), Int64, false},
	{int32(0), Int32, false},
	{true, Bool, false},
	{newNode(withGraph(NewGraph()), withType(Float64), withOp(newEBOByType(addOpType, Float64, Float64))), Float64, false},

	{[]int{0}, nil, true},
}

func TestInferType(t *testing.T) {
	for _, itts := range inferTypeTests {
		t0, err := inferType(itts.expr)
		switch {
		case itts.err && err == nil:
			t.Errorf("Expected an error in infering type of %T", itts.expr)
		case !itts.err && err != nil:
			t.Errorf("Error while inferring type of %T: %v", itts.expr, err)
		}

		if itts.err {
			continue
		}
		assert.True(t, itts.correct.Eq(t0), "%v != %v", t0, itts.correct)
	}

	// way out there stuff
	g := NewGraph()
	n := newNode(withGraph(g), withOp(newEBOByType(addOpType, Float64, Float64)), withChildren(Nodes{newNode(withGraph(g), WithName("a"), withType(Float64)), newNode(withGraph(g), WithName("b"), withType(Float64))}))
	t0, err := inferType(n)
	if err != nil {
		t.Errorf("Special Case #1: %v", err)
	}
	t.Logf("t0: %v", t0)
}

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
		{Float64, Float64, false},
		{newTensorType(1, Float64), Float64, false},
		{hm.NewTypeVar("a", hm.WithInstance(Float64)), Float64, false},
		{hm.NewTypeVar("a", hm.WithInstance(newTensorType(1, Float64))), Float64, false},
		{hm.NewTypeVar("a", hm.WithInstance(hm.NewTypeVar("b", hm.WithInstance(malformed{})))), Float64, true},

		// bad shit
		{hm.NewTypeVar("a"), MAXDTYPE, true},
		{newTensorType(1, hm.NewTypeVar("a")), MAXDTYPE, true},
	}
}
