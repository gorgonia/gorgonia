package gorgonia

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPrune(t *testing.T) {
	assert := assert.New(t)

	t.Log("Empty type variable")
	tv1 := new(typeVariable)
	pruned := prune(tv1)

	assert.Equal(tv1, pruned)

	t.Log("Concrete instance")
	t1 := Float64
	tv1.instance = t1
	pruned = prune(tv1)

	assert.Equal(t1, pruned)

	t.Log("I N C E P T I O N - many leveles of typeVariables")
	tv2 := new(typeVariable)
	tv2.instance = t1
	tv1.instance = tv2
	pruned = prune(tv1)

	assert.Equal(t1, pruned)
}

func TestTypeEq(t *testing.T) {
	t.Log("Simple type variable equality")

	tv1 := newTypeVariable("a")
	tv2 := newTypeVariable("b")
	tv3 := newTypeVariable("a")

	if eq := typeEq(tv1, tv2); eq {
		t.Errorf("Expected tv1 and tv2 to be not equal")
	}

	if eq := typeEq(tv1, tv3); !eq {
		t.Errorf("Expected tv1 and tv3 to be equal")
	}

	t.Log("Simple Dtype equality")
	t1 := Float32
	t2 := Float32
	t3 := Float64

	if eq := typeEq(t1, t2); !eq {
		t.Errorf("Expected t1 and t2 to be equal")
	}

	if eq := typeEq(t1, t3); eq {
		t.Errorf("Expected t1 and t3 to be not equal (Float32 != Float64)")
	}

	t.Log("Testing tensor type constructors - simple tests: different dims, different concrete dtypes")
	tt1 := newTensorType(1, Float64)
	tt2 := newTensorType(2, Float64)
	tt3 := newTensorType(1, Float32)
	tt4 := newTensorType(1, Float64)

	if eq := typeEq(tt1, tt2); eq {
		t.Errorf("Expected tt1 and tt2 to not be equal (different dims)")
	}

	if eq := typeEq(tt1, tt3); eq {
		t.Errorf("Expected tt1 and tt2 to not be equal (different concrete dtypes)")
	}

	if eq := typeEq(tt1, tt4); !eq {
		t.Errorf("Expected tt1 and tt4 to be equal")
	}

	t.Log("Testing more complex examples: with typeVariables instead of concrete dtypes")
	tv2.instance = Float64
	tv3.instance = Float64
	tt5 := newTensorType(1, tv1) // no instance - Tensor a
	tt6 := newTensorType(1, tv2) // has instance - Tensor b; b = Float64
	tt7 := newTensorType(1, tv3) // has instance - Tensor a; a = Float64

	if eq := typeEq(tt1, tt5); eq {
		t.Errorf("Expected tt1 and tt5 to be not equal")
	}

	if eq := typeEq(tt1, tt6); !eq {
		t.Errorf("Expected tt1 and tt6 to be equal")
	}

	if eq := typeEq(tt5, tt7); eq {
		t.Errorf("Expected tt5 and tt7 to be NOT be equal (one is a concrete `Tensor Float64` and the other is `Tensor a`")
	}

	if eq := typeEq(tt1, tt7); !eq {
		t.Errorf("Expected tt1 and tt7 to be equal (they resolve ultimately to the same type)")
		t.Errorf("tt1: %v", tt1)
		t.Errorf("tt7: %v", tt7)
	}
}

func TestUnify(t *testing.T) {
	// the tests only ever really tests things that have been seen while running the type system
	assert := assert.New(t)
	var t1, t2 Type

	t.Log("t1: typeVariable; t2: Float64")
	tv1 := newTypeVariable("a")
	t1 = tv1
	t2 = Float64
	if err := unify(t1, t2); err != nil {
		t.Errorf("Error occured: %v", err)
	}
	assert.Equal(t2, tv1.instance)

	//******************************

	t.Log("t1: typeVariable; t2: typeVariable. Both have the same name, no instance")
	tv1 = newTypeVariable("a")
	tv2 := newTypeVariable("a")

	t1 = tv1
	t2 = tv2

	if err := unify(t1, t2); err == nil {
		t.Errorf("There should have been a error")
	}
	assert.Equal(tv2.instance, tv1.instance)
	assert.Equal(tv2.constraints, tv1.constraints)

	//******************************

	t.Log("t1: typeVariable; t2: typeVariable. Different names")
	tv1 = newTypeVariable("a")
	tv2 = newTypeVariable("b")

	t1 = tv1
	t2 = tv2

	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}
	assert.Equal(tv2, tv1.instance)
	assert.Equal(tv2.constraints, tv1.constraints)

	//******************************

	t.Log("t1: Dtype, t2: Dtype")
	t1 = Float64
	t2 = Float64

	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}

	t2 = Float32
	if err := unify(t1, t2); err == nil {
		t.Errorf("differing Dtypes should yield error")
	}

	//******************************

	t.Log("t1: Dtype, t2: typeVar")
	tv2.instance = nil
	t2 = tv2

	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}
	assert.Equal(t1, tv2.instance)

	tv1.instance = Float64
	t2 = t1
	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}
	assert.Equal(t1, tv1.instance)

	//******************************

	t.Log("t1: Tensor a; t2: a")
	a := newTypeVariable("a")
	t1 = &TensorType{of: a}
	tv1.instance = nil
	t2 = tv1

	if err := unify(t1, t2); err == nil {
		t.Errorf("Expected a recursive unification error (because both have the same typeVariable)")
	}

	//******************************

	t.Log("t1: Tensor a; t2: b")
	t1 = &TensorType{of: a}
	tv2.instance = nil
	t2 = tv2

	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}
	assert.Equal(t1, tv2.instance)

	//******************************

	// other common cases
	t.Log("t1: Tensor a; t2: Tensor Float64")
	t1 = &TensorType{of: a}
	t2 = &TensorType{of: Float64}

	if err := unify(t1, t2); err != nil {
		t.Errorf("Error: %v", err)
	}
	if eq := typeEq(t1, t2); !eq {
		t.Errorf("The results should be equal")
	}

}
