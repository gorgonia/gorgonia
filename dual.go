package gorgonia

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type dualValue struct {
	Value
	d Value // the derivative wrt to each input
}

func (dv *dualValue) SetDeriv(d Value) error {
	if t, ok := d.(tensor.Tensor); ok && t.IsScalar() {
		d, _ = anyToScalar(t.ScalarValue())
	}
	dv.d = d

	return dv.sanity()
}

func (dv *dualValue) SetValue(v Value) error {
	dv.Value = v
	return dv.sanity()
}

func (dv *dualValue) Clone() (retVal interface{}, err error) {
	var v, d Value
	if v, err = CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	if dv.d != nil {
		if d, err = CloneValue(dv.d); err != nil {
			return nil, errors.Wrap(err, cloneFail)
		}
	}

	dv2 := borrowDV()
	dv2.Value = v
	dv2.d = d
	retVal = dv2
	return
}

func (dv *dualValue) Type() hm.Type       { return TypeOf(dv.Value) }
func (dv *dualValue) Dtype() tensor.Dtype { return dv.Value.Dtype() }

func (dv *dualValue) ValueEq(a Value) bool {
	switch at := a.(type) {
	case *dualValue:
		if at == dv {
			return true
		}
		veq := ValueEq(at.Value, dv.Value)
		deq := ValueEq(at.d, dv.d)
		return veq && deq
	// case Value:
	// 	return ValueEq(at, dv.Value)
	default:
		return false
	}
}

func (dv *dualValue) String() string {
	return fmt.Sprintf("%#+v", dv.Value)
}

func (dv *dualValue) sanity() error {
	// check that d and v are the same type

	// dvv := typeCheckTypeOf(dv.Value)
	// dvd := typeCheckTypeOf(dv.d)
	// if !dvv.Eq(dvd) {
	// 	return errors.Errorf("DualValues do not have the same types: %v and %v", dvv, dvd)
	// }
	// ReturnType(dvv)
	// ReturnType(dvd)

	// TODO: check that the shapes are the same

	return nil
}

// clones the dualValue and zeroes out the ndarrays
func (dv *dualValue) clone0() (retVal *dualValue, err error) {
	var v, d Value
	if v, err = CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	if d, err = CloneValue(dv.d); err != nil {
		return nil, errors.Wrap(err, cloneFail)
	}

	v = ZeroValue(v)
	d = ZeroValue(d)

	dv2 := borrowDV()
	dv2.Value = v
	dv2.d = d
	retVal = dv2
	return
}

// the derivative of a constant is zero.
//
// The original implementation was to have a constantDualValue type. This would lead to waaay less allocations of matrices
// but as it turns out, as I waws working, the constants turn out to be not so constant afterall.
// Is this a problem with the graph that leads to derivation of constant values? I don't quite know. TO CHECK
func constantDV(val Value) *dualValue {
	enterLogScope()
	defer leaveLogScope()

	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val

	var err error
	if retVal.d, err = CloneValue(val); err != nil {
		panic(err)
	}

	retVal.d = ZeroValue(retVal.d)
	return retVal
}

// the derivative of x is 1.
func variableDV(val Value) *dualValue {
	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val

	switch v := val.(type) {
	case Scalar:
		retVal.d = one(v.Dtype())
	case tensor.Tensor:
		shp := v.Shape()
		dt := v.Dtype()
		retVal.d = tensor.Ones(dt, shp...)
	default:
		panic(fmt.Sprintf("%v(%T) not handled yet", v, v))
	}

	return retVal
}

// monadic unit() function. This unit() function will allocate a Value for dv.d
// this is useful for forward mode autodiff
func dvUnit(v Value) *dualValue {
	enterLogScope()
	defer leaveLogScope()

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

// dvUnitManaged does dvUnit for values whose memories are manually managed
func dvUnitManaged(v Value, op *ExternalOp) (*dualValue, error) {
	if op.Device == CPU {
		return dvUnit(v), nil
	}

	if dv, ok := v.(*dualValue); ok {
		return dv, nil
	}

	retVal := borrowDV()
	retVal.Value = v

	s := v.Shape()
	dt := v.Dtype()
	memsize := calcMemSize(dt, s)
	// allocate on device
	mem, err := op.Get(op.Device, memsize)
	if err != nil {
		return nil, err
	}

	d, err := makeValueFromMem(TypeOf(v), s, mem)
	if err != nil {
		return nil, err
	}
	retVal.d = d

	return retVal, nil
}

func dvUnitVarManaged(v Value, op *ExternalOp) (*dualValue, error) {
	dv, err := dvUnitManaged(v, op)
	if err != nil {
		return dv, err
	}

	switch d := dv.d.(type) {
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
	case *F64:
		*d = F64(1)
	case *F32:
		*d = F32(1)
	case *I:
		*d = I(1)
	case *I64:
		*d = I64(1)
	case *I32:
		*d = I32(1)
	case *U8:
		*d = U8(1)
	case *B:
		*d = B(true)
	default:
		return dv, errors.Errorf("Unhandeled type: %T", d)
	}
	return dv, nil
}

// helper to unpack from []*dualValue
func idValue(inputs []*dualValue) (retVals []Value) {
	retVals = make([]Value, len(inputs))
	for i, input := range inputs {
		retVals[i] = input.Value
	}
	return
}

// dvBind applies an op to the inputs, and returns a *dualValue
func dvBind(op Op, inputs []*dualValue) (retVal *dualValue, err error) {
	enterLogScope()
	defer leaveLogScope()

	vals := idValue(inputs)

	var ret Value
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
func dvBindVar(op Op, inputs []*dualValue) (retVal *dualValue, err error) {
	vals := idValue(inputs)

	var ret Value
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, opDoFail)
	}
	if o, ok := op.(*ExternalOp); ok {
		return dvUnitVarManaged(ret, o)
	}
	return dvUnitVar(ret), nil
}

//TODO test vecvecdot divBind0

// doesn't alloc a dualValue, and reuses whatever that is there, and zeroes out the deriv
func dvBind0(op Op, retVal *dualValue, inputs []*dualValue) (err error) {
	prealloc := retVal.Value
	vals := idValue(inputs)

	var ret Value
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

	retVal.SetDeriv(ZeroValue(retVal.d))
	return
}

func dvBindVar0(op Op, retVal *dualValue, inputs []*dualValue) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	var ret Value
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

	switch v := retVal.d.(type) {
	case Scalar:
		retVal.d = one(v.Dtype())
	case tensor.Tensor:
		switch v.Dtype() {
		case tensor.Float64:
			err = v.Memset(float64(1))
		case tensor.Float32:
			err = v.Memset(float32(1))
		}
		retVal.d = v
	default:
		err = errors.Errorf(nyiTypeFail, "dvBindVar0", retVal.d)
	}
	return
}
