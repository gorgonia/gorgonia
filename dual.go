package gorgonia

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// the derivative of a constant is zero.
//
// The original implementation was to have a constantDualValue type. This would lead to waaay less allocations of matrices
// but as it turns out, as I waws working, the constants turn out to be not so constant afterall.
// Is this a problem with the graph that leads to derivation of constant values? I don't quite know. TO CHECK
func constantDV(val value.Value) *value.DualValue {
	enterLogScope()
	defer leaveLogScope()

	// retVal := &value.DualValue{Value: val}
	retVal := value.BorrowDV()
	retVal.Value = val

	var err error
	if retVal.D, err = value.CloneValue(val); err != nil {
		panic(err)
	}

	retVal.D = value.ZeroValue(retVal.D)
	return retVal
}

// the derivative of x is 1.
func variableDV(val value.Value) *value.DualValue {
	// retVal := &value.DualValue{Value: val}
	retVal := value.BorrowDV()
	retVal.Value = val

	switch v := val.(type) {
	case value.Scalar:
		retVal.D = value.One(v.Dtype())
	case tensor.Tensor:
		shp := v.Shape()
		dt := v.Dtype()
		retVal.D = tensor.Ones(dt, shp...)
	default:
		panic(fmt.Sprintf("%v(%T) not handled yet", v, v))
	}

	return retVal
}

// monadic unit() function. This unit() function will allocate a value.Value for dv.d
// this is useful for forward mode autodiff
func dvUnit(v value.Value) *value.DualValue {
	enterLogScope()
	defer leaveLogScope()

	if dv, ok := v.(*value.DualValue); ok {
		return dv
	}
	return constantDV(v)
}

func dvUnitVar(v value.Value) *value.DualValue {
	if dv, ok := v.(*value.DualValue); ok {
		return dv
	}
	return variableDV(v)
}

// no alloc is done. It'll just return a *value.DualValue with nil as the dv.d
func dvUnit0(v value.Value) *value.DualValue {
	if dv, ok := v.(*value.DualValue); ok {
		return dv
	}

	retVal := value.BorrowDV()
	retVal.Value = v

	return retVal
}

// dvUnitManaged does dvUnit for values whose memories are manually managed
func dvUnitManaged(v value.Value, op *ExternalOp) (*value.DualValue, error) {
	if op.Device == execution.CPU {
		return dvUnit(v), nil
	}

	if dv, ok := v.(*value.DualValue); ok {
		return dv, nil
	}

	retVal := value.BorrowDV()
	retVal.Value = v

	s := v.Shape()
	dt := v.Dtype()
	memsize := calcMemSize(dt, s)
	// allocate on device
	mem, err := op.Get(op.Device, memsize)
	if err != nil {
		return nil, err
	}

	d, err := makeValueFromMem(value.TypeOf(v), s, mem)
	if err != nil {
		return nil, err
	}
	retVal.D = d

	return retVal, nil
}

func dvUnitVarManaged(v value.Value, op *ExternalOp) (*value.DualValue, error) {
	dv, err := dvUnitManaged(v, op)
	if err != nil {
		return dv, err
	}

	switch d := dv.D.(type) {
	case tensor.Tensor:
		dt := d.Dtype()
		switch dt {
		case tensor.Float64:
			d.Memset(1.0)
		case tensor.Float32:
			d.Memset(float32(1))
		case tensor.Bool:
			d.Memset(true)
		default:
			return dv, errors.Errorf("Unhandled dtype: %v", dt)
		}
	case *value.F64:
		*d = value.F64(1)
	case *value.F32:
		*d = value.F32(1)
	case *value.I:
		*d = value.I(1)
	case *value.I64:
		*d = value.I64(1)
	case *value.I32:
		*d = value.I32(1)
	case *value.U8:
		*d = value.U8(1)
	case *value.B:
		*d = value.B(true)
	default:
		return dv, errors.Errorf("Unhandeled type: %T", d)
	}
	return dv, nil
}

// helper to unpack from []*value.DualValue
func idValue(inputs []*value.DualValue) (retVals []value.Value) {
	retVals = make([]value.Value, len(inputs))
	for i, input := range inputs {
		retVals[i] = input.Value
	}
	return
}

// dvBind applies an op to the inputs, and returns a *value.DualValue
func dvBind(op Op, inputs []*value.DualValue) (retVal *value.DualValue, err error) {
	enterLogScope()
	defer leaveLogScope()

	vals := idValue(inputs)

	var ret value.Value
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, opDoFail)
	}
	if o, ok := op.(*ExternalOp); ok {
		return dvUnitManaged(ret, o)
	}
	return dvUnit(ret), nil
}

// dvBindVar returns a dvUnitVar instead of dvUnit (which zeroes the derivative).
// The default derivative of a variable wrt itself is 1 (dx/dx == 1)
func dvBindVar(op Op, inputs []*value.DualValue) (retVal *value.DualValue, err error) {
	vals := idValue(inputs)

	var ret value.Value
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, opDoFail)
	}
	if o, ok := op.(*ExternalOp); ok {
		return dvUnitVarManaged(ret, o)
	}
	return dvUnitVar(ret), nil
}

//TODO test vecvecdot divBind0

// doesn't alloc a value.DualValue, and reuses whatever that is there, and zeroes out the deriv
func dvBind0(op Op, retVal *value.DualValue, inputs []*value.DualValue) (err error) {
	prealloc := retVal.Value
	vals := idValue(inputs)

	var ret value.Value
	if pd, ok := op.(UsePreallocDoer); ok {
		if ret, err = pd.UsePreallocDo(prealloc, vals...); err == nil {
			goto next
		}
	}
	if ret, err = op.Do(vals...); err != nil {
		return errors.Wrap(err, opDoFail)
	}

next:
	if err != nil {
		return
	}

	if err = retVal.SetValue(ret); err != nil {
		return
	}

	retVal.SetDeriv(value.ZeroValue(retVal.D))
	return
}

func dvBindVar0(op Op, retVal *value.DualValue, inputs []*value.DualValue) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	var ret value.Value
	if pd, ok := op.(UsePreallocDoer); ok {
		ret, err = pd.UsePreallocDo(prealloc, vals...)
	} else {
		if ret, err = op.Do(vals...); err != nil {
			return errors.Wrap(err, opDoFail)
		}
	}

	if err != nil {
		return errors.Wrapf(err, opDoFail)
	}

	if err = retVal.SetValue(ret); err != nil {
		return errors.Wrap(err, "Failed at setting the value")
	}

	switch v := retVal.D.(type) {
	case value.Scalar:
		retVal.D = value.One(v.Dtype())
	case tensor.Tensor:
		switch v.Dtype() {
		case tensor.Float64:
			err = v.Memset(float64(1))
		case tensor.Float32:
			err = v.Memset(float32(1))
		}
		retVal.D = v
	default:
		err = errors.Errorf(nyiTypeFail, "dvBindVar0", retVal.D)
	}
	return
}
