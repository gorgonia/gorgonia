package tensor

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCS_Basics(t *testing.T) {
	assert := assert.New(t)
	xs0 := []int{1, 2, 6, 8}
	ys0 := []int{1, 2, 1, 6}
	xs1 := []int{1, 2, 6, 8}
	ys1 := []int{1, 2, 1, 6}
	vals0 := []float64{3, 1, 4, 1}
	vals1 := []float64{3, 1, 4, 1}

	var T0, T1 *CS
	var d0, d1 *Dense
	var dp0, dp1 *Dense
	var err error
	fails := func() {
		CSCFromCoord(Shape{7, 6}, xs0, ys0 , vals0)
	}
	assert.Panics(fails)

	// Test CSC
	T0 = CSCFromCoord(Shape{9, 7},xs0, ys0,  vals0)
	d0 = T0.Dense()
	T0.T()
	dp0 = T0.Dense()
	T0.UT() // untranspose as Materialize() will be called below

	// Test CSR
	fails = func() {
		CSRFromCoord(Shape{7, 6},xs1, ys1,  vals1)
	}
	T1 = CSRFromCoord(Shape{9, 7}, xs1, ys1,  vals1)
	d1 = T1.Dense()
	T1.T()
	dp1 = T1.Dense()
	T1.UT()

	t.Logf("%v %v", T0.indptr, T0.indices)
	t.Logf("%v %v", T1.indptr, T1.indices)

	assert.True(d0.Eq(d1), "%+#v\n %+#v\n", d0, d1)
	assert.True(dp0.Eq(dp1))
	assert.True(T1.Eq(T1))
	assert.False(T0.Eq(T1))

	// At
	var got interface{}
	correct := float64(3.0)
	if got, err = T0.At(1, 1); err != nil {
		t.Error(err)
	}
	if got.(float64) != correct {
		t.Errorf("Expected %v. Got %v - T0[1,1]", correct, got)
	}
	if got, err = T1.At(1, 1); err != nil {
		t.Error(err)
	}
	if got.(float64) != correct {
		t.Errorf("Expected %v. Got %v - T1[1,1]", correct, got)
	}

	correct = 0.0
	if got, err = T0.At(3, 3); err != nil {
		t.Error(err)
	}
	if got.(float64) != correct {
		t.Errorf("Expected %v. Got %v - T0[3,3]", correct, got)
	}

	if got, err = T1.At(3, 3); err != nil {
		t.Error(err)
	}
	if got.(float64) != correct {
		t.Errorf("Expected %v. Got %v - T1[3,3]", correct, got)
	}

	// Test clone
	T2 := T0.Clone()
	assert.True(T0.Eq(T2))

	// Scalar representation
	assert.False(T0.IsScalar())
	fails = func() {
		T0.ScalarValue()
	}
	assert.Panics(fails)

	assert.False(T0.IsView())
	assert.True(d0.Eq(T0.Materialize()), "d0: \n%+#v\nMat: \n%#+v\n", d0, T0.Materialize())

	assert.Equal(len(vals0), T0.NonZeroes())

}
