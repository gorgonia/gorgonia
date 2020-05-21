package dual

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/execution"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/internal/memutils"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

type Op interface {
	Do(...values.Value) (values.Value, error)
}

type ExternalOp interface {
	Op
	Get(device execution.Device, size int64) (tensor.Memory, error)
	Device() execution.Device
}

type UsePreallocDoer interface {
	UsePreallocDo(prealloc values.Value, inputs ...values.Value) (values.Value, error)
}

// Dual represents a dual value. In this instance, a dual value usually holds the value and a gradient value.
type Dual struct {
	values.Value
	d values.Value
}

// SetDeriv sets the derivative value
func (dv *Dual) SetDeriv(d values.Value) error {
	if t, ok := d.(tensor.Tensor); ok && t.IsScalar() {
		d, _ = values.AnyToScalar(t.ScalarValue())
	}
	dv.d = d

	return dv.sanity()
}

// SetValue sets the value.
func (dv *Dual) SetValue(v values.Value) error {
	dv.Value = v
	return dv.sanity()
}

func (dv *Dual) SetEngine(e tensor.Engine) {
	values.SetEngine(dv.Value, e)
	values.SetEngine(dv.d, e)
}

// Clone clones a Dual
func (dv *Dual) Clone() (retVal interface{}, err error) {
	var v, d values.Value
	if v, err = values.CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, gerrors.CloneFail)
	}

	if dv.d != nil {
		if d, err = values.CloneValue(dv.d); err != nil {
			return nil, errors.Wrap(err, gerrors.CloneFail)
		}
	}

	dv2 := borrowDV()
	dv2.Value = v
	dv2.d = d
	retVal = dv2
	return
}

func (dv *Dual) Type() hm.Type       { return values.TypeOf(dv.Value) }
func (dv *Dual) Dtype() tensor.Dtype { return dv.Value.Dtype() }

func (dv *Dual) ValueEq(a values.Value) bool {
	switch at := a.(type) {
	case *Dual:
		if at == dv {
			return true
		}
		veq := values.ValueEq(at.Value, dv.Value)
		deq := values.ValueEq(at.d, dv.d)
		return veq && deq
	// case Value:
	// 	return ValueEq(at, dv.Value)
	default:
		return false
	}
}

func (dv *Dual) String() string {
	return fmt.Sprintf("%#+v", dv.Value)
}

func (dv *Dual) sanity() error {
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
func (dv *Dual) clone0() (retVal *Dual, err error) {
	var v, d values.Value
	if v, err = values.CloneValue(dv.Value); err != nil {
		return nil, errors.Wrap(err, gerrors.CloneFail)
	}

	if d, err = values.CloneValue(dv.d); err != nil {
		return nil, errors.Wrap(err, gerrors.CloneFail)
	}

	v = values.ZeroValue(v)
	d = values.ZeroValue(d)

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
func constantDV(val values.Value) *Dual {
	enterLogScope()
	defer leaveLogScope()

	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val

	var err error
	if retVal.d, err = values.CloneValue(val); err != nil {
		panic(err)
	}

	retVal.d = values.ZeroValue(retVal.d)
	return retVal
}

// the derivative of x is 1.
func variableDV(val values.Value) *Dual {
	// retVal := &dualValue{Value: val}
	retVal := borrowDV()
	retVal.Value = val

	switch v := val.(type) {
	case values.Scalar:
		retVal.d = values.One(v.Dtype())
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
func dvUnit(v values.Value) *Dual {
	enterLogScope()
	defer leaveLogScope()

	if dv, ok := v.(*Dual); ok {
		return dv
	}
	return constantDV(v)
}

func dvUnitVar(v values.Value) *Dual {
	if dv, ok := v.(*Dual); ok {
		return dv
	}
	return variableDV(v)
}

// no alloc is done. It'll just return a *Dual with nil as the dv.d
func dvUnit0(v values.Value) *Dual {
	if dv, ok := v.(*Dual); ok {
		return dv
	}

	retVal := borrowDV()
	retVal.Value = v

	return retVal
}

// dvUnitManaged does dvUnit for values whose memories are manually managed
func dvUnitManaged(v values.Value, op ExternalOp) (*Dual, error) {
	if op.Device() == execution.CPU {
		return dvUnit(v), nil
	}

	if dv, ok := v.(*Dual); ok {
		return dv, nil
	}

	retVal := borrowDV()
	retVal.Value = v

	s := v.Shape()
	dt := v.Dtype()
	memsize := memutils.MemSize(dt, s)
	// allocate on device
	mem, err := op.Get(op.Device(), memsize)
	if err != nil {
		return nil, err
	}

	d, err := values.MakeFromMem(values.TypeOf(v), s, mem)
	if err != nil {
		return nil, err
	}
	retVal.d = d

	return retVal, nil
}

func dvUnitVarManaged(v values.Value, op ExternalOp) (*Dual, error) {
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
	case values.Oner:
		// important: we need to put this case before the OneValuer case.
		//
		// this is especially important when we use GPU, as all we get are borrowed references (to use Rust's terms)
		// So we can only mutate the pointers, and not create new Values, which OneValuer does.
		//
		// So if a type implements Oner and OneValuer, Oner should have precedence.
		d.One()
	case values.OneValuer:
		dv.d = d.OneValue()
	default:
		return dv, errors.Errorf("Unhandeled type: %T", d)
	}
	return dv, nil
}

// helper to unpack from []*Dual
func idValue(inputs []*Dual) (retVals []values.Value) {
	retVals = make([]values.Value, len(inputs))
	for i, input := range inputs {
		retVals[i] = input.Value
	}
	return
}

// dvBind applies an op to the inputs, and returns a *Dual
func dvBind(op Op, inputs []*Dual) (retVal *Dual, err error) {
	enterLogScope()
	defer leaveLogScope()

	vals := idValue(inputs)

	var ret values.Value
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	if o, ok := op.(ExternalOp); ok {
		return dvUnitManaged(ret, o)
	}
	return dvUnit(ret), nil
}

// dvBindVar returns a dvUnitVar instead of dvUnit (which zeroes the derivative).
// The default derivative of a variable wrt itself is 1 (dx/dx == 1)
func dvBindVar(op Op, inputs []*Dual) (retVal *Dual, err error) {
	vals := idValue(inputs)

	var ret values.Value
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	if o, ok := op.(ExternalOp); ok {
		return dvUnitVarManaged(ret, o)
	}
	return dvUnitVar(ret), nil
}

//TODO test vecvecdot divBind0

// doesn't alloc a dualValue, and reuses whatever that is there, and zeroes out the deriv
func dvBind0(op Op, retVal *Dual, inputs []*Dual) (err error) {
	prealloc := retVal.Value
	vals := idValue(inputs)

	var ret values.Value
	if pd, ok := op.(UsePreallocDoer); ok {
		if ret, err = pd.UsePreallocDo(prealloc, vals...); err == nil {
			goto next
		}
	}
	if ret, err = op.Do(vals...); err != nil {
		return errors.Wrap(err, gerrors.OpDoFail)
	}

next:
	if err != nil {
		return
	}

	if err = retVal.SetValue(ret); err != nil {
		return
	}

	retVal.SetDeriv(values.ZeroValue(retVal.d))
	return
}

func dvBindVar0(op Op, retVal *Dual, inputs []*Dual) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	var ret values.Value
	if pd, ok := op.(UsePreallocDoer); ok {
		ret, err = pd.UsePreallocDo(prealloc, vals...)
	} else {
		if ret, err = op.Do(vals...); err != nil {
			return errors.Wrap(err, gerrors.OpDoFail)
		}
	}

	if err != nil {
		return errors.Wrapf(err, gerrors.OpDoFail)
	}

	if err = retVal.SetValue(ret); err != nil {
		return errors.Wrap(err, "Failed at setting the value")
	}

	switch v := retVal.d.(type) {
	case values.Scalar:
		retVal.d = values.One(v.Dtype())
	case tensor.Tensor:
		switch v.Dtype() {
		case tensor.Float64:
			err = v.Memset(float64(1))
		case tensor.Float32:
			err = v.Memset(float32(1))
		}
		retVal.d = v
	default:
		err = errors.Errorf(gerrors.NYITypeFail, "dvBindVar0", retVal.d)
	}
	return
}
