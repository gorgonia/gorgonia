package dual

import (
	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
	"gorgonia.org/tensor"
)

// New creates a new *Dual[DT].
//
// Its behaviour is a little different from what Gophers might be used to.
// If the value passed in is itself a *Dual[DT], then it returns itself.
// Otherwise it returns a new *Dual[DT].
//
// New assumes that the value that is passed in is to be treated as a
// constant, hence the Deriv() is going to be 0. If it is desired that a *Dual[DT]
// is created for a variable (i.e. the default derivative is 1), then use NewVar.
//
// Additional notes: this is analogous to `unit` or `pure` in Haskell.
// A *Dual[DT] is a monadic representation of a dual value.
func New[DT tensor.Num](v values.Value[DT]) *Dual[DT] {
	// formerly known as dvUnit
	if dv, ok := v.(*Dual[DT]); ok {
		return dv
	}
	return constantDV[DT](v)
}

// NewVar creates a new *Dual[DT] assuming that the provided value is to be treated as a variable.
//
// Other behaviours from New() is preserved.
func NewVar[DT tensor.Num](v values.Value[DT]) *Dual[DT] {
	// formerly known as dvUnitVar
	if dv, ok := v.(*Dual[DT]); ok {
		return dv
	}
	return variableDV[DT](v)
}

// BindVar performs the operation on the inputs. The result is a *Dual[DT,T] that assumes that it is a variable value.
func BindVar[DT tensor.Num](op Op[DT], inputs ...*Dual[DT]) (retVal *Dual[DT], err error) {
	var ret values.Value[DT]
	if ret, err = op(idValue(inputs)...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	return NewVar[DT](ret), nil
}

// Bind0 performs the operation using a preallocated *Dual[DT,T]. The resulting deriv is not set.
func Bind0[DT tensor.Num](op PreallocOp[DT], retVal *Dual[DT], inputs ...*Dual[DT]) (*Dual[DT], error) {
	prealloc := retVal.Value

	ret, err := op(prealloc, idValue(inputs)...)
	if err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	if err = retVal.SetValue(ret); err != nil {
		return nil, errors.Wrap(err, "Unable to SetValue in Bind0")
	}
	err = retVal.SetDeriv(retVal.d)
	return retVal, err
}

// Bind performs the operation on the inputs. The result is a *Dual[DT,T] with the d value set by the provided DualOp.
func Bind[DT tensor.Num](op DualOp[DT], inputs ...*Dual[DT]) (retVal *Dual[DT], err error) {
	var ret values.Value[DT]
	if ret, err = op.Do(idValue[DT](inputs)...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	var deriv values.Value[DT]
	if deriv, err = op.Dual(inputs...); err != nil {
		return nil, errors.Wrap(err, "Unable to perform dual bindings")
	}
	retVal = New[DT](ret)
	err = retVal.SetDeriv(deriv) // TODO: copy values? or Set?
	return

}

// LiftVar transforms a Op into a function that takes the equivalent in *Dual[DT,T]s.
func LiftVar[DT tensor.Num](op Op[DT]) func(values ...*Dual[DT]) (*Dual[DT], error) {
	return func(inputs ...*Dual[DT]) (retVal *Dual[DT], err error) {
		return BindVar(op, inputs...)
	}
}

// Lift transforms a DualOp into a function that takes the equivalent in *Dual[DT,T]s
func Lift[DT tensor.Num](op DualOp[DT]) func(values ...*Dual[DT]) (*Dual[DT], error) {
	return func(inputs ...*Dual[DT]) (*Dual[DT], error) { return Bind(op, inputs...) }
}

// All checks that all values.Value[DT] are *Dual[DT,T]. It returns a list of *Dual[DT,T], and a bool indicating if it's all *Dual[DT,T].
// If not, the list will be empty.
func All[DT tensor.Num](vals ...values.Value[DT]) ([]*Dual[DT], bool) {
	retVal := make([]*Dual[DT], len(vals))
	for i := range vals {
		d, ok := vals[i].(*Dual[DT])
		if !ok {
			return nil, false
		}
		retVal[i] = d
	}
	return retVal, true
}

// NewAlike is a function that clones the given *Dual[DT,T]. However, the values and deriv are zeroed out.
func NewAlike[DT tensor.Num](a *Dual[DT]) (retVal *Dual[DT], err error) {
	return a.clone0()
}
