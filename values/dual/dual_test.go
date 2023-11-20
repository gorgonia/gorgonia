package dual

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	stdeng "gorgonia.org/tensor/engines"
	"gorgonia.org/tensor/scalar"
)

// This example shows the various ways to create a *Dual.
// New is a way to turn a standard Value into a Dual value.
//
// By default New creates a dual value for a Value that is considered constant.
// If a dual value for a variable is desired, use NewVar.
func Example_new() {
	v, _ := values.AnyToScalar(1.0)
	c, _ := values.AnyToScalar(3.1415)

	// The usual ways to create a new *Dual.

	// New() creates a new *Dual, assuming that the input value represents a constatn
	d0 := New[float64](c)
	fmt.Printf("c: %v, d0 %#v\n", d0, d0)

	// if a *Dual gets passed into New(), it returns the input arg.
	d1 := New[float64](d0)
	fmt.Printf("d1 == d0: %t\n", d0 == d1)

	// Where New() was meant for values representing a constant, NewVar() is meant for values representing a variable.
	// Observe that the derivatives are different.
	d2 := New[float64](v)
	d3 := NewVar[float64](v)
	fmt.Printf("v: %v d2 %#v, d3 %#v\n", v, d2, d3)

	// NewVar() is like New() - if you pass in a *Dual, it returns the input.
	d4 := NewVar[float64](d0)
	fmt.Printf("d4 == d0: %t\n", d4 == d0)

	// using new() is another way to construct a *Dual.
	// In this scenario, you would want to use the .SetValue() and .SetDeriv() methods to set a value.
	//
	// This is generally not used as often.
	d5 := new(Dual[float64])
	d5.SetValue(v)
	d5.SetDeriv(c)
	fmt.Printf("d5: %#v\n", d5)

	// New() and NewVar() works on all Values, that includes tensor.Tensor
	a := dense.New[float64](tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	a2 := New[float64](a)
	a3 := NewVar[float64](a)
	fmt.Printf("a:\n%va2:%#v\na3:%#v", a, a2, a3)

	// Output:
	// c: 3.1415, d0 {3.1415 | 0}
	// d1 == d0: true
	// v: 1 d2 {1 | 0}, d3 {1 | 1}
	// d4 == d0: true
	// d5: {1 | 3.1415}
	// a:
	// ⎡1  2⎤
	// ⎣3  4⎦
	// a2:{
	// value:
	// ⎡1  2⎤
	// ⎣3  4⎦
	// ---
	// deriv:
	// ⎡0  0⎤
	// ⎣0  0⎦
	// }
	// a3:{
	// value:
	// ⎡1  2⎤
	// ⎣3  4⎦
	// ---
	// deriv:
	// ⎡1  1⎤
	// ⎣1  1⎦
	// }

}

// This example shows how to use *Dual.
// See  the related exampled of Lift
func Example_bindVar() {
	times := func(vals ...scalar.Scalar[float64]) (scalar.Scalar[float64], error) {
		if len(vals) != 2 {
			return scalar.Scalar[float64]{}, errors.New("Expected 2")
		}

		a := vals[0].Data()[0]
		b := vals[1].Data()[0]
		retVal, _ := values.AnyToScalar(a * b)
		return retVal, nil
	}

	three, _ := values.AnyToScalar(3.0)
	two, _ := values.AnyToScalar(2.0)
	five, _ := values.AnyToScalar(5.0)

	threetimestwo, err := BindVar(times, NewVar[float64](three), NewVar[float64](two))
	fmt.Printf("Using BindVar: 3 × 2 = %#v. Err: %v\n", threetimestwo, err)

	preallocTimes := func(prealloc scalar.Scalar[float64], inputs ...scalar.Scalar[float64]) (scalar.Scalar[float64], error) {
		ret, err := times(inputs...)
		return ret, err
	}
	custom := new(Dual[float64])
	custom.SetValue(three)
	custom.SetDeriv(five)
	threetimestwo, err = Bind0(preallocTimes, custom, NewVar[float64](three), NewVar[float64](two))
	fmt.Printf("Using Bind0: 3 × 2 = %#v. Err: %v\n", threetimestwo, err)

	// Output:
	// Using BindVar: 3 × 2 = {6 | 1}. Err: <nil>
	// Using Bind0: 3 × 2 = {6 | 5}. Err: <nil>
}

// TestBindVar  tests the failure modes not covered in ExampleBind
func TestBindVar(t *testing.T) {
	times := func(vals ...scalar.Scalar[float64]) (scalar.Scalar[float64], error) {
		if len(vals) != 2 {
			return scalar.Scalar[float64]{}, errors.New("Expected 2")
		}

		a := vals[0].V
		b := vals[1].V
		retVal, _ := values.AnyToScalar(a * b)
		return retVal, nil
	}

	three, _ := values.AnyToScalar(3.0)
	two, _ := values.AnyToScalar(2.0)
	five, _ := values.AnyToScalar(5.0)

	if _, err := BindVar(times, New[float64](three), New[float64](two), New[float64](three)); err == nil {
		t.Errorf("Expected an error.")
	}

	preallocTimes := func(prealloc scalar.Scalar[float64], inputs ...scalar.Scalar[float64]) (scalar.Scalar[float64], error) {
		_, err := times(inputs...)
		return prealloc, err
	}
	custom := new(Dual[float64])
	custom.SetValue(three)
	custom.SetDeriv(five)
	if _, err := Bind0(preallocTimes, custom, NewVar[float64](three), NewVar[float64](two), New[float64](two)); err == nil {
		t.Errorf("Expected an error.")
	}

}

func Example_lift() {
	times := func(vals ...scalar.Scalar[float64]) (scalar.Scalar[float64], error) {
		if len(vals) != 2 {
			return scalar.Scalar[float64]{}, errors.New("Expected 2")
		}

		a := vals[0].V
		b := vals[1].V
		retVal, _ := values.AnyToScalar(a * b)
		return retVal, nil
	}

	three, _ := values.AnyToScalar(3.0)
	two, _ := values.AnyToScalar(2.0)
	liftedTimes := LiftVar[float64](times)
	threetimestwo, err := liftedTimes(NewVar[float64](three), NewVar[float64](two))
	fmt.Printf("Using lifttedTimes %T: 3 × 2 = %#v. Err: %v\n", liftedTimes, threetimestwo, err)

	// Output:
	// Using lifttedTimes func(...*dual.Dual) (*dual.Dual, error): 3 × 2 = {6 | 1}. Err: <nil>

}

func TestClone(t *testing.T) {
	assert := assert.New(t)

	f, _ := values.AnyToScalar(3.14)
	fds := New[float64](f)
	fds2 := fds.Clone()

	assert.True(fds.ValueEq(fds2))
	if fds == fds2 {
		t.Error("Cloned values should never be the same pointer!")
	}

	T := dense.New[float32](tensor.WithShape(3, 4), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}))
	Tds := NewVar[float32](T)
	Tds2 := Tds.Clone()
	assert.True(Tds.ValueEq(Tds2))

}

func TestNewAlike(t *testing.T) {
	assert := assert.New(t)
	f, _ := values.AnyToScalar(3.14)
	fds := NewVar[float64](f)
	fds2, err := NewAlike(fds)

	assert.Nil(err)
	assert.Equal(0.0, fds2.Value.Data())
	assert.Equal(0.0, fds2.Deriv().Data())
	assert.False(fds.ValueEq(fds2), "Should be different values: fds %v | fds2 %v", fds, fds2)
	if fds == fds2 {
		t.Error("Cloned values should never be the same pointer!")
	}

	T := dense.New[float32](tensor.WithShape(3, 4), tensor.WithBacking([]float32{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}))
	Tds := NewVar[float32](T)
	Tds2, err := NewAlike(Tds)
	assert.Nil(err)
	assert.False(Tds.ValueEq(Tds2), "Should be different values: Tds %v | Tds2 %v", Tds, Tds2)
}

func TestSetEngine(t *testing.T) {
	f, _ := values.AnyToScalar(3.14)
	fds := NewVar[float64](f)
	fds.SetEngine(stdeng.StdEng[float64, scalar.Scalar[float64]]{}) // should not panic
}
