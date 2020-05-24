package dual_test

import (
	"fmt"

	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
)

// mul is a Multiplication operation. It also implements DualOp
type mul func(vals ...values.Value) (values.Value, error)

func (m mul) Do(vals ...values.Value) (values.Value, error) { return m(vals...) }

func (m mul) Dual(ds ...*dual.Dual) (values.Value, error) {
	a := valToFloat64(ds[0].Value)
	c := valToFloat64(ds[1].Value)
	bɛ := valToFloat64(ds[0].Deriv())
	dɛ := valToFloat64(ds[1].Deriv())
	return values.NewF64((a*bɛ + c*dɛ)), nil
}

// valToFloat64 is a helper function for this example
func valToFloat64(v values.Value) float64 {
	return float64(*(v.(*values.F64)))
}

// singlesMul is the kernel of the op (i.e. the actual algorithm for multiplicatoin)
func singlesMul(vals ...values.Value) (values.Value, error) {
	a := valToFloat64(vals[0])
	b := valToFloat64(vals[1])
	return values.NewF64(a * b), nil
}

func Mul(vals ...values.Value) (values.Value, error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}

	a := *(ds[0].Value.(*values.F64))
	bɛ := *(ds[0].Deriv().(*values.F64))
	c := *(ds[1].Value.(*values.F64))
	dɛ := *(ds[1].Deriv().(*values.F64))

	e := a * c
	f := a*dɛ + c*bɛ

	retVal := new(dual.Dual)
	if err := retVal.SetValue(&e); err != nil {
		return nil, err
	}
	if err := retVal.SetDeriv(&f); err != nil {
		return nil, err
	}
	return retVal, nil
}

func Mul2(vals ...values.Value) (values.Value, error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}
	return dual.Bind(mul(singlesMul), ds...)
}

func MulRevAD(vals ...values.Value) (values.Value, error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}
	retVal, err := dual.BindVar(singlesMul, ds...)
	if err != nil {
		return nil, err
	}

	// These are incomplete accumulations.
	// In reality, they should not be just Set, but needs to be accumulated
	if err := ds[0].SetDeriv(ds[1].Value); err != nil {
		return nil, err
	}
	if err := ds[1].SetDeriv(ds[0].Value); err != nil {
		return nil, err
	}
	return retVal, err
}

func Example() {
	two := values.NewF64(2)
	three := values.NewF64(3)

	twods := dual.NewVar(two)
	threeds := dual.NewVar(three)

	// Vanilla operations
	six0, err := Mul(two, three)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", two, three, six0)

	// Usual Dual numbers
	sixds, err := Mul(twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds)

	// Usual Dual Numbers, but using Bind()
	sixds2, err := Mul2(twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds2)

	// Usual Dual Numbers, using Lift
	liftedMul := dual.Lift(mul(singlesMul))
	sixds3, err := liftedMul(twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds3)

	// Reusing Dual Number structure to perform reverse mode automatic differentiation.
	sixdsRev, err := MulRevAD(twods, threeds)
	if err != nil {
		fmt.Printf("wee %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixdsRev)

	// Output:
	// 2 × 3 = 6
	// {2 | 1} × {3 | 1} = {6 | 5}
	// {2 | 1} × {3 | 1} = {6 | 5}
	// {2 | 1} × {3 | 1} = {6 | 5}
	// {2 | 3} × {3 | 2} = {6 | 1}

}
