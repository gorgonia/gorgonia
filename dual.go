package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor"
)

type dualValue struct {
	Value
	d Value // the derivative wrt to each input
}

func (dv *dualValue) SetDeriv(d Value) error {
	dv.d = d
	return dv.sanity()
}

func (dv *dualValue) SetValue(v Value) error {
	dv.Value = v
	return dv.sanity()
}

func (dv *dualValue) sanity() error {
	// check that d and v are the same type

	if !typeEq(dv.Value.Type(), dv.d.Type()) {
		return NewError(AutoDiffError, "DualValues do not have the same types")
	}

	// TODO: check that the shapes are the same

	return nil
}

func (dv *dualValue) clone() (retVal Value, err error) {
	var v, d Value
	if v, err = dv.Value.clone(); err != nil {
		return
	}

	if d, err = dv.d.clone(); err != nil {
		return
	}

	dv2 := borrowDV()
	dv2.Value = v
	dv2.d = d
	retVal = dv2
	return
}

// clones the dualValue and zeroes out the ndarrays
func (dv *dualValue) clone0() (retVal *dualValue, err error) {
	var v, d Value
	if v, err = dv.Value.clone(); err != nil {
		return
	}

	if d, err = dv.d.clone(); err != nil {
		return
	}

	switch vt := v.(type) {
	case Tensor:
		vt.Tensor.Zero()
	case Scalar:
		switch vt.t {
		case Float64:
			vt.v = 0.0
		case Float32:
			vt.v = float32(0.0)
		case Int:
			vt.v = 0
		case Int32:
			vt.v = int32(0)
		case Int64:
			vt.v = int64(0)
		case Bool:
			vt.v = false
		}
	}

	switch vt := d.(type) {
	case Tensor:
		vt.Tensor.Zero()
	case Scalar:
		switch vt.t {
		case Float64:
			vt.v = 0.0
		case Float32:
			vt.v = float32(0.0)
		case Int:
			vt.v = 0
		case Int32:
			vt.v = int32(0)
		case Int64:
			vt.v = int64(0)
		case Bool:
			vt.v = false
		}
	}

	dv2 := borrowDV()
	dv2.Value = v
	dv2.d = d
	retVal = dv2
	return
}

func (dv *dualValue) String() string {
	return fmt.Sprintf("%#+v", dv.Value)
}

// the derivative of a constant is zero.
//
// The original implementation was to have a constantDualValue type. This would lead to waaay less allocations of matrices
// but as it turns out, as I waws working, the constants turn out to be not so constant afterall.
// Is this a problem with the graph that leads to derivation of constant values? I don't quite know. TO CHECK
func constantDV(val Value) *dualValue {
	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val
	var d Value
	switch v := val.(type) {
	case Tensor:
		dt := v.Tensor.Dtype()
		shp := v.Shape()
		t := tensor.Zeroes(dt, shp...)
		d = Tensor{Tensor: t}
		// switch dtypeToDtype(dt) {
		// case Float64:
		// 	d = NewScalarValue(float64(0.0))
		// case Float32:
		// 	d = NewScalarValue(float32(0.0))
		// case Int:
		// 	d = NewScalarValue(int(0))
		// default:
		// 	panic(fmt.Sprintf("Scalar of type %v not yet handled", dt))
		// }

	case Scalar:
		switch v.t {
		case Float64:
			d = NewScalarValue(float64(0.0))
		case Float32:
			d = NewScalarValue(float32(0.0))
		case Int:
			d = NewScalarValue(int(0))
		default:
			panic(fmt.Sprintf("Scalar of type %v not yet handled", v.t))
		}
	}
	retVal.d = d
	return retVal
}

// the derivative of x is 1.
func variableDV(val Value) *dualValue {
	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val

	var d Value
	switch v := val.(type) {
	case Tensor:
		shp := v.Shape()
		dt := v.Tensor.Dtype()
		t := tensor.Ones(dt, shp...)
		// tt := prune(v.Type()).(*TensorType)
		// d = newTensorValue(tt, t)
		d = Tensor{Tensor: t}
	case Scalar:
		switch v.t {
		case Float64:
			d = NewScalarValue(float64(1.0))
		case Float32:
			d = NewScalarValue(float32(1.0))
		case Int:
			d = NewScalarValue(int(1))
		default:
			panic(fmt.Sprintf("Scalar of type %v not yet handled", v.t))
		}
	}
	retVal.d = d
	return retVal
}

// monadic unit() function. This unit() function will allocate a Value for dv.d
// this is useful for forward mode autodiff
func dvUnit(v Value) *dualValue {
	if dv, ok := v.(*dualValue); ok {
		return dv
	}

	return constantDV(v)
}

func dvUnitVar(v Value) *dualValue {
	if dv, ok := v.(*dualValue); ok {
		return dv
	}
	return variableDV(v)
}

// no alloc is done. It'll just return a *dualValue with nil as the dv.d
func dvUnit0(v Value) *dualValue {
	if dv, ok := v.(*dualValue); ok {
		return dv
	}

	retVal := borrowDV()
	retVal.Value = v

	return retVal
}

// helper to unpack from []*dualValue
func idValue(inputs []*dualValue) (retVals []Value) {
	retVals = make([]Value, len(inputs))
	for i, input := range inputs {
		retVals[i] = input.Value
	}
	return
}

func dvBind(op Op, inputs []*dualValue) (retVal *dualValue, err error) {
	vals := idValue(inputs)

	var ret Value
	if ret, err = op.Do(vals...); err == nil {
		retVal = dvUnit(ret)
	}
	return
}

// dvBindVar returns a dvUnitVar instead of dvUnit (which zeroes the derivative).
// The default derivative of a variable wrt itself is 1 (dx/dx == 1)
func dvBindVar(op Op, inputs []*dualValue) (retVal *dualValue, err error) {
	vals := idValue(inputs)

	var ret Value
	if ret, err = op.Do(vals...); err == nil {
		retVal = dvUnitVar(ret)
	}
	return
}

// doesn't alloc a dualValue, and reuses whatever that is there, and zeroes out the deriv
func dvBind0(op Op, retVal *dualValue, inputs []*dualValue) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	if pd, ok := op.(UsePreallocDoer); ok {
		_, err = pd.UsePreallocDo(prealloc, vals...)
	} else {
		var ret Value
		if ret, err = op.Do(vals...); err != nil {
			return
		}
		err = retVal.SetValue(ret)
	}

	retVal.SetDeriv(retVal.d.zero())
	return
}

func dvBindVar0(op Op, retVal *dualValue, inputs []*dualValue) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	if pd, ok := op.(UsePreallocDoer); ok {
		_, err = pd.UsePreallocDo(prealloc, vals...)
	} else {
		var ret Value
		if ret, err = op.Do(vals...); err != nil {
			return
		}
		err = retVal.SetValue(ret)
	}

	if err != nil {
		return
	}

	var d Value
	switch v := retVal.d.(type) {
	case Tensor:
		switch v.Dtype() {
		case Float64:
			err = v.SetAll(float64(1.0))
		case Float32:
			err = v.SetAll(float32(1.0))
		case Int:
			err = v.SetAll(int(1))
		default:
			panic(fmt.Sprintf("Tensor of type %v not yet handled", v.Dtype()))
		}
		d = v
	case Scalar:
		switch v.t {
		case Float64:
			d = NewScalarValue(float64(1.0))
		case Float32:
			d = NewScalarValue(float32(1.0))
		case Int:
			d = NewScalarValue(int(1))
		default:
			panic(fmt.Sprintf("Scalar of type %v not yet handled", v.t))
		}
	}
	retVal.d = d
	return
}
