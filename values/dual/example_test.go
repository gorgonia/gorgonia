package dual_test

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/gorgonia/values/dual"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	"gorgonia.org/tensor/scalar"
)

// mul is a Multiplication operation. It also implements DualOp
type mul[DT tensor.Num, T tensor.Tensor[DT, T]] func(vals ...T) (T, error)

func (m mul[DT, T]) Do(vals ...T) (T, error) { return m(vals...) }

func (m mul[DT, T]) Dual(ds ...*dual.Dual[DT, T]) (retVal T, err error) {
	a := ds[0].Value()
	b := ds[1].Value()
	da := ds[0].Deriv()
	db := ds[1].Deriv()
	ada, err := singlesMul[DT](a, da)
	if err != nil {
		return retVal, err
	}
	bdb, err := singlesMul[DT](b, db)
	if err != nil {
		return retVal, err
	}
	return singlesAdd[DT](ada, bdb)
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
func singlesMul[DT tensor.Num, T tensor.Tensor[DT, T]](vals ...T) (retVal T, err error) {
	if len(vals) != 2 {
		return retVal, errors.Errorf("Expected only two inputs")
	}
	data0 := vals[0].Data()
	data1 := vals[1].Data()
	if len(data0) != 1 || len(data1) != 1 {
		return retVal, errors.Errorf("Expected only two inputs")
	}
	retData := make([]DT, len(data0))
	for i, v := range data0 {
		retData[i] = v * data1[i]
	}
	return vals[0].Alike(tensor.WithBacking(retData), tensor.WithShape(vals[0].Shape()...)), nil
}

func singlesAdd[DT tensor.Num, T tensor.Tensor[DT, T]](vals ...T) (retVal T, err error) {
	if len(vals) != 2 {
		return retVal, errors.Errorf("Expected only two inputs")
	}
	data0 := vals[0].Data()
	data1 := vals[1].Data()
	if len(data0) != 1 || len(data1) != 1 {
		return retVal, errors.Errorf("Expected only two inputs")
	}
	retData := make([]DT, len(data0))
	for i, v := range data0 {
		retData[i] = v + data1[i]
	}
	return vals[0].Alike(tensor.WithBacking(retData), tensor.WithShape(vals[0].Shape()...)), nil
}

func valMul[DT tensor.Num](vals ...values.Value[DT]) (values.Value[DT], error) {
	if len(vals) != 2 {
		return nil, errors.Errorf("Expected only two inputs")
	}
	data0 := vals[0].Data()
	data1 := vals[1].Data()
	if len(data0) != 1 || len(data1) != 1 {
		return nil, errors.Errorf("Expected only two inputs")
	}
	retData := make([]DT, len(data0))
	for i, v := range data0 {
		retData[i] = v * data1[i]
	}

	return dense.New[DT](tensor.WithBacking(retData), tensor.WithShape(vals[0].Shape()...)), nil
}

func Mul[DT tensor.Num, T tensor.Tensor[DT, T]](vals ...values.Value[DT]) (retVal values.Value[DT], err error) {
	ds, ok := dual.All[DT, T](vals...)
	if !ok {
		return valMul[DT](vals...)
	}

	a := ds[0].Value()
	b := ds[1].Value()
	c, err := singlesMul[DT, T](a, b)
	if err != nil {
		return retVal, err
	}

	da := ds[0].Deriv()
	db := ds[1].Deriv()

	x, err := singlesMul[DT, T](a, db)
	if err != nil {
		return retVal, err
	}

	y, err := singlesMul[DT, T](b, da)
	if err != nil {
		return retVal, err
	}
	z, err := singlesAdd[DT, T](x, y)
	if err != nil {
		return retVal, err
	}

	ret := new(dual.Dual[DT, T])

	if err := ret.SetValue(c); err != nil {
		return nil, err
	}
	if err := ret.SetDeriv(z); err != nil {
		return nil, err
	}
	return ret, nil
}

func Mul2[DT tensor.Num, T tensor.Tensor[DT, T]](vals ...values.Value[DT]) (values.Value[DT], error) {
	ds, ok := dual.All[DT, T](vals...)
	if !ok {
		return valMul(vals...)
	}
	return dual.Bind[DT, T](mul[DT, T](singlesMul[DT, T]), ds...)
}

func MulRevAD[DT tensor.Num, T tensor.Tensor[DT, T]](vals ...values.Value[DT]) (values.Value[DT], error) {
	ds, ok := dual.All[DT, T](vals...)
	if !ok {
		return valMul(vals...)
	}
	retVal, err := dual.BindVar[DT, T](singlesMul[DT, T], ds...)
	if err != nil {
		return nil, err
	}

	// These are incomplete accumulations.
	// In reality, they should not be just Set, but needs to be accumulated
	if err := ds[0].SetDeriv(ds[1].Value()); err != nil {
		return nil, err
	}
	if err := ds[1].SetDeriv(ds[0].Value()); err != nil {
		return nil, err
	}
	return retVal, err
}

func Example() {
	two, _ := values.AnyToScalar[float64](2)
	three, _ := values.AnyToScalar[float64](3)

	twods := dual.NewVar[float64, *dense.Dense[float64]](two)
	threeds := dual.NewVar[float64, *dense.Dense[float64]](three)

	// Vanilla operations
	six0, err := Mul[float64, *dense.Dense[float64]](two, three)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", two, three, six0)

	// Usual Dual numbers
	sixds, err := Mul[float64, *dense.Dense[float64]](twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds)

	// Usual Dual Numbers, but using Bind()
	sixds2, err := Mul2[float64, *dense.Dense[float64]](twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds2)

	// Usual Dual Numbers, using Lift
	liftedMul := dual.Lift[float64, *dense.Dense[float64]](mul[float64, *dense.Dense[float64]](singlesMul[float64, *dense.Dense[float64]]))
	sixds3, err := liftedMul(twods, threeds)
	if err != nil {
		fmt.Printf("err: %v\n", err)
	}
	fmt.Printf("%#v × %#v = %#v\n", twods, threeds, sixds3)

	// Reusing Dual Number structure to perform reverse mode automatic differentiation.
	sixdsRev, err := MulRevAD[float64, *dense.Dense[float64]](twods, threeds)
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
