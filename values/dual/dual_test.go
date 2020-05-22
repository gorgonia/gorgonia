package dual

import (
	"errors"
	"fmt"
	"testing"

	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// This example shows the various ways to create a *Dual.
// New is a way to turn a standard Value into a Dual value.
//
// By default New creates a dual value for a Value that is considered constant.
// If a dual value for a variable is desired, use NewVar.
func Example_new() {
	v := values.NewF64(1)
	c := values.NewF64(3.1415)

	// The usual ways to create a new *Dual.

	// New() creates a new *Dual, assuming that the input value represents a constatn
	d0 := New(c)
	fmt.Printf("c: %v, d0 %#v\n", c, d0)

	// if a *Dual gets passed into New(), it returns the input arg.
	d1 := New(d0)
	fmt.Printf("d1 == d0: %t\n", d0 == d1)

	// Where New() was meant for values representing a constant, NewVar() is meant for values representing a variable.
	// Observe that the derivatives are different.
	d2 := New(v)
	d3 := NewVar(v)
	fmt.Printf("v: %v d2 %#v, d3 %#v\n", v, d2, d3)

	// NewVar() is like New() - if you pass in a *Dual, it returns the input.
	d4 := NewVar(d0)
	fmt.Printf("d4 == d0: %t\n", d4 == d0)

	// using new() is another way to construct a *Dual.
	// In this scenario, you would want to use the .SetValue() and .SetDeriv() methods to set a value.
	//
	// This is generally not used as often.
	d5 := new(Dual)
	d5.SetValue(v)
	d5.SetDeriv(c)
	fmt.Printf("d5: %#v\n", d5)

	// New() and NewVar() works on all Values, that includes tensor.Tensor
	a := tensor.New(tensor.WithShape(2, 2), tensor.WithBacking([]float32{1, 2, 3, 4}))
	a2 := New(a)
	a3 := NewVar(a)
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
func Example_bind() {
	times := func(vals ...values.Value) (values.Value, error) {
		if len(vals) != 2 {
			return nil, errors.New("Expected 2")
		}

		a := float64(*(vals[0].(*values.F64)))
		b := float64(*(vals[1].(*values.F64)))
		retVal := values.NewF64(a * b)
		return retVal, nil
	}

	three := values.NewF64(3)
	two := values.NewF64(2)
	five := values.NewF64(5)

	threetimestwo, err := BindVar(times, NewVar(three), NewVar(two))
	fmt.Printf("Using BindVar: 3 × 2 = %#v. Err: %v\n", threetimestwo, err)

	threetimestwo, err = Bind(times, NewVar(three), NewVar(two))
	fmt.Printf("Using Bind: 3 × 2 = %#v. Err: %v\n", threetimestwo, err)

	preallocTimes := func(prealloc values.Value, inputs ...values.Value) (values.Value, error) {
		p := prealloc.(*values.F64)
		ret, err := times(inputs...)
		*p = *(ret.(*values.F64))
		return p, err
	}
	custom := new(Dual)
	custom.SetValue(three)
	custom.SetDeriv(five)
	threetimestwo, err = Bind0(preallocTimes, custom, NewVar(three), NewVar(two))
	fmt.Printf("Using Bind0: 3 × 2 = %#v. Err: %v\n", threetimestwo, err)

	// Output:
	// Using BindVar: 3 × 2 = {6 | 1}. Err: <nil>
	// Using Bind: 3 × 2 = {6 | 0}. Err: <nil>
	// Using Bind0: 3 × 2 = {6 | 5}. Err: <nil>
}

// TestBind  tests the failure modes not covered in ExampleBind
func TestBind(t *testing.T) {
	times := func(vals ...values.Value) (values.Value, error) {
		if len(vals) != 2 {
			return nil, errors.New("Expected 2")
		}

		a := float64(*(vals[0].(*values.F64)))
		b := float64(*(vals[1].(*values.F64)))
		retVal := values.NewF64(a * b)
		return retVal, nil
	}

	three := values.NewF64(3)
	two := values.NewF64(2)
	five := values.NewF64(5)

	if _, err := Bind(times, NewVar(three), New(two), New(three)); err == nil {
		t.Errorf("Expected an error.")
	}

	if _, err := BindVar(times, New(three), New(two), New(three)); err == nil {
		t.Errorf("Expected an error.")
	}

	preallocTimes := func(prealloc values.Value, inputs ...values.Value) (values.Value, error) {
		p := prealloc.(*values.F64)
		_, err := times(inputs...)
		return p, err
	}
	custom := new(Dual)
	custom.SetValue(three)
	custom.SetDeriv(five)
	if _, err := Bind0(preallocTimes, custom, NewVar(three), NewVar(two), New(two)); err == nil {
		t.Errorf("Expected an error.")
	}

}

func Example_lift() {
	times := func(vals ...values.Value) (values.Value, error) {
		if len(vals) != 2 {
			return nil, errors.New("Expected 2")
		}

		a := float64(*(vals[0].(*values.F64)))
		b := float64(*(vals[1].(*values.F64)))
		retVal := values.NewF64(a * b)
		return retVal, nil
	}
	three := values.NewF64(3)
	two := values.NewF64(2)
	liftedTimes := Lift(times)
	threetimestwo, err := liftedTimes(NewVar(three), NewVar(two))
	fmt.Printf("Using lifttedTimes %T: 3 × 2 = %#v. Err: %v\n", liftedTimes, threetimestwo, err)

	// Output:
	// Using lifttedTimes func(...*dual.Dual) (*dual.Dual, error): 3 × 2 = {6 | 0}. Err: <nil>

}
