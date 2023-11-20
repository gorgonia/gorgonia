package dual_test

import (
	"fmt"

	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	"gorgonia.org/tensor/scalar"
)

// mul is a Multiplication operation. It also implements DualOp
type mul[DT tensor.Num] func(vals ...values.Value[DT]) (values.Value[DT], error)

func (m mul[DT]) Do(vals ...values.Value[DT]) (values.Value[DT], error) { return m(vals...) }

func (m mul[DT]) Dual(ds ...*dual.Dual[DT]) (values.Value[DT], error) {
	a := valueToGoScalar[DT](ds[0].Value)
	c := valueToGoScalar[DT](ds[1].Value)
	bɛ := valueToGoScalar[DT](ds[0].Deriv())
	dɛ := valueToGoScalar[DT](ds[1].Deriv())
	return scalar.S[DT]((a*bɛ + c*dɛ)), nil
}

// valueToGoScalar is a helper function for this example
func valueToGoScalar[DT tensor.Num](v values.Value[DT]) DT {
	switch v := any(v).(type) {
	case scalar.Scalar[DT]:
		return v.V
	case tensor.Basic[DT]:
		return v.Data()[0]
	}
	panic("Unreachable")

}

// singlesMul is the kernel of the op (i.e. the actual algorithm for multiplicatoin)
func singlesMul[DT tensor.Num, T values.Value[DT]](vals ...T) (T, error) {
	var z T
	switch any(z).(type) {
	case scalar.Scalar[DT]:
		return scalar.S[DT](valueToGoScalar(vals[0]) * valueToGoScalar(vals[1])), nil
	case *dense.Dense[DT]:
		return dense.Mul(vals[0].(*dense.Dense[DT]), vals[1].(*dense.Dense[DT])), nil
	}
	panic("Unreachable")
}

func Mul[DT tensor.Num](vals ...values.Value[DT]) (values.Value[DT], error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}

	a := valueToGoScalar(ds[0].Value)
	bɛ := valueToGoScalar(ds[0].Deriv())
	c := valueToGoScalar(ds[1].Value)
	dɛ := valueToGoScalar(ds[1].Deriv())

	e := a * c
	f := a*dɛ + c*bɛ

	retVal := new(dual.Dual[DT])
	if err := retVal.SetValue(scalar.S[DT](e)); err != nil {
		return nil, err
	}
	if err := retVal.SetDeriv(scalar.S[DT](f)); err != nil {
		return nil, err
	}
	return retVal, nil
}

func Mul2[DT tensor.Num](vals ...values.Value[DT]) (values.Value[DT], error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}
	return dual.Bind[float64](mul(singlesMul[DT]), ds...)
}

func MulRevAD[DT tensor.Num](vals ...values.Value[DT]) (values.Value[DT], error) {
	ds, ok := dual.All(vals...)
	if !ok {
		return singlesMul(vals...)
	}
	retVal, err := dual.BindVar[DT](singlesMul[DT], ds...)
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
	two := scalar.S[float64](2)
	three := scalar.S[float64](3)

	twods := dual.NewVar[float64](two)
	threeds := dual.NewVar[float64](three)

	// Vanilla operations
	six0, err := Mul[float64](two, three)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", two, three, six0)

	// Usual Dual numbers
	sixds, err := Mul[float64](twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds)

	// Usual Dual Numbers, but using Bind()
	sixds2, err := Mul2[float64](twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds2)

	// Usual Dual Numbers, using Lift
	liftedMul := dual.Lift[float64](mul[float64](singlesMul[float64]))
	sixds3, err := liftedMul(twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds3)

	// Reusing Dual Number structure to perform reverse mode automatic differentiation.
	sixdsRev, err := MulRevAD[float64](twods, threeds)
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
