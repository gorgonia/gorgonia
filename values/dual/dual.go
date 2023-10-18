package dual

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
)

//var _ datatypes.Tensor[float64] = &Dual[float64]{}

// Op is a function that takes an arbitrary number of Values and returns a Value
type Op[DT tensor.Num, T Value[DT, T]] func(vals ...T) (T, error)

// PreallocOp is a function that has the return value specified and preallocated, then takes an arbitrary number of Values and returns a Value.
type PreallocOp[DT tensor.Num, T Value[DT, T]] func(prealloc T, inputs ...T) (T, error)

// DualOp is any op that can perform its forwards operation on *Dual.
type DualOp[DT tensor.Num, T Value[DT, T]] interface {
	Do(vals ...T) (T, error)
	Dual(vals ...*Dual[DT, T]) (T, error)
}

type Value[DT tensor.Num, T values.Value[DT]] interface {
	values.Value[DT]
	values.Cloner[T]
	comparable
}

// Dual represents a dual value. In this instance, a dual value usually holds the value and a gradient value.
type Dual[DT tensor.Num, T Value[DT, T]] struct {
	v T
	d T
}

/* implement values.Value because you can no longer embed a generic type into a struct. Grr */

// tensor.Desc

func (dv *Dual[DT, T]) Dtype() dtype.Dtype  { return dv.v.Dtype() }
func (dv *Dual[DT, T]) Shape() shapes.Shape { return dv.v.Shape() }
func (dv *Dual[DT, T]) Strides() []int      { return dv.v.Strides() }
func (dv *Dual[DT, T]) Dims() int           { return dv.v.Dims() }
func (dv *Dual[DT, T]) Size() int           { return dv.v.Size() }
func (dv *Dual[DT, T]) Info() *tensor.AP    { return dv.v.Info() }

// tensor.DataSizer

func (dv *Dual[DT, T]) DataSize() int { return dv.v.DataSize() }

// implementing tensor.Memory

func (dv *Dual[DT, T]) MemSize() uintptr { return dv.v.MemSize() }
func (dv *Dual[DT, T]) Uintptr() uintptr { return dv.v.Uintptr() }

// implementing tensor.Engineer

func (dv *Dual[DT, T]) Engine() tensor.Engine { return dv.v.Engine() }

// implementing tensor.RawAccessor[DT]

func (dv *Dual[DT, T]) Data() []DT { return dv.v.Data() }

// implementing tensor.ValueSetter[DT]

func (dv *Dual[DT, T]) SetAt(v DT, coord ...int) error { return dv.v.SetAt(v, coord...) }
func (dv *Dual[DT, T]) Memset(v DT) error              { return dv.v.Memset(v) }
func (dv *Dual[DT, T]) Zero()                          { dv.v.Zero() }

// SetDeriv sets the derivative value
func (dv *Dual[DT, T]) SetDeriv(d T) error {
	dv.d = d

	return dv.sanity()
}

// SetValue sets the value.
func (dv *Dual[DT, T]) SetValue(v T) error {
	dv.v = v
	return dv.sanity()
}

// SetEngine sets the engine.
func (dv *Dual[DT, T]) SetEngine(e tensor.Engine) {
	values.SetEngine[DT](dv.v, e)
	values.SetEngine[DT](dv.d, e)
}

// Deriv returns the derivative value.
func (dv *Dual[DT, T]) Deriv() values.Value[DT] { return dv.d }

// Clone clones a *Dual[DT,T].
func (dv *Dual[DT, T]) Clone() *Dual[DT, T] {
	var v, d T
	v = values.Clone(dv.v)

	var z T
	if dv.d != z {
		d = values.Clone(dv.d)
	}

	dv2 := new(Dual[DT, T])
	dv2.v = v
	dv2.d = d
	return dv2
}

// Type returns the type of the values in the *Dual[DT,T].
func (dv *Dual[DT, T]) Type() hm.Type { return values.TypeOf(dv.v) }

// ValueEq implements values.Value[DT]Eqer, which states that Values can be compared.
func (dv *Dual[DT, T]) ValueEq(a values.Value[DT]) bool {
	switch at := a.(type) {
	case *Dual[DT, T]:
		if at == dv {
			return true
		}
		veq := values.ValueEq[DT](at.v, dv.v)
		deq := values.ValueEq[DT](at.d, dv.d)
		return veq && deq
	// case Value:
	// 	return ValueEq(at, dv.Value)
	default:
		return false
	}
}

func (dv *Dual[DT, T]) String() string { return fmt.Sprintf("%#+v", dv.v) }

func (dv *Dual[DT, T]) Format(s fmt.State, c rune) {
	isScalar := dv.v.Shape().Eq(shapes.ScalarShape())
	if s.Flag('#') {
		if isScalar {
			fmt.Fprintf(s, "{%v | %v}", dv.v, dv.d)
		} else {
			fmt.Fprintf(s, "{\nvalue:\n%v---\nderiv:\n%v}", dv.v, dv.d)
		}
		return
	}

	fmt.Fprintf(s, "%v", dv.v)
}

// CopyFrom copies the values from a values.Value[DT] to the first value of the *Dual[DT,T]. The deriv is untouched.
func (dv *Dual[DT, T]) CopyFrom(src interface{}) error {
	if v, ok := src.(values.Value[DT]); ok {
		_, err := values.Copy[DT](dv.v, v)
		return err
	}
	return errors.Errorf("Unable to CopyFrom %T", src)
}

func (dv *Dual[DT, T]) sanity() error {
	// check that d and v are the same type

	// dvv := typeCheckTypeOf(dv.v)
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
func (dv *Dual[DT, T]) clone0() (retVal *Dual[DT, T], err error) {
	var v, d T
	v = values.Clone(dv.v)
	d = values.Clone(dv.d)

	v = values.ZeroValue[DT](v).(T)
	d = values.ZeroValue[DT](d).(T)

	dv2 := new(Dual[DT, T])
	dv2.v = v
	dv2.d = d
	retVal = dv2
	return
}

// the derivative of a constant is zero.
//
// The original implementation was to have a constantDualValue type. This would lead to waaay less allocations of matrices
// but as it turns out, as I waws working, the constants turn out to be not so constant afterall.
// Is this a problem with the graph that leads to derivation of constant values? I don't quite know. TO CHECK
func constantDV[DT tensor.Num, T Value[DT, T]](val T) *Dual[DT, T] {
	enterLogScope()
	defer leaveLogScope()

	// retVal := &dualValue{Value: val}
	retVal := new(Dual[DT, T])
	retVal.v = val

	retVal.d = values.Clone(val)
	retVal.d = values.ZeroValue[DT](retVal.d).(T)
	return retVal
}

// the derivative of x is 1.
func variableDV[DT tensor.Num, T Value[DT, T]](val values.Value[DT]) *Dual[DT, T] {
	// retVal := &dualValue{Value: val}
	retVal := new(Dual[DT, T])
	retVal.v = val.(T)

	switch v := val.(type) {
	// case scalar.Scalar[DT]:
	// 	retVal.d = values.One[DT]()
	case tensor.Basic[DT]:
		shp := v.Shape()
		//dt := v.Dtype()
		retVal.d = any(dense.Ones[DT](shp...)).(T)
	default:
		panic(fmt.Sprintf("%v(%T) not handled yet", v, v))
	}

	return retVal
}

/*
// dvUnitManaged does dvUnit for values whose memories are manually managed
func dvUnitManaged(v values.Value[DT], op ExternalOp) (*Dual[DT,T], error) {
	if op.Device() == execution.CPU {
		return New(v), nil
	}

	if dv, ok := v.(*Dual[DT,T]); ok {
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


func dvUnitVarManaged(v values.Value[DT], op ExternalOp) (*Dual[DT,T], error) {
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
*/

// helper to unpack from []*Dual[DT,T]
func idValue[DT tensor.Num, T Value[DT, T]](inputs []*Dual[DT, T]) (retVals []T) {
	retVals = make([]T, len(inputs))
	for i, input := range inputs {
		retVals[i] = input.v
	}
	return
}

/*

// dvBind applies an op to the inputs, and returns a *Dual[DT,T]
func dvBind(op Op, inputs []*Dual[DT,T]) (retVal *Dual[DT,T], err error) {
	enterLogScope()
	defer leaveLogScope()

	vals := idValue(inputs)

	var ret values.Value[DT]
	if ret, err = op.Do(vals...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	if o, ok := op.(ExternalOp); ok {
		return dvUnitManaged(ret, o)
	}
	return New(ret), nil
}

// dvBindVar returns a dvUnitVar instead of dvUnit (which zeroes the derivative).
// The default derivative of a variable wrt itself is 1 (dx/dx == 1)
func dvBindVar(op Op, inputs []*Dual[DT,T]) (retVal *Dual[DT,T], err error) {
	vals := idValue(inputs)

	var ret values.Value[DT]
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
func dvBind0(op Op, retVal *Dual[DT,T], inputs []*Dual[DT,T]) (err error) {
	prealloc := retVal.Value
	vals := idValue(inputs)

	var ret values.Value[DT]
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

func dvBindVar0(op Op, retVal *Dual[DT,T], inputs []*Dual[DT,T]) (err error) {
	prealloc := retVal.Value

	vals := idValue(inputs)

	var ret values.Value[DT]
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
*/
