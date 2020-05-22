package dual

import (
	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/values"
)

// New creates a new *Dual.
//
// Its behaviour is a little different from what Gophers might be used to.
// If the value passed in is itself a *Dual, then it returns itself.
// Otherwise it returns a new *Dual.
//
// New assumes that the value that is passed in is to be treated as a
// constant, hence the Deriv() is going to be 0. If it is desired that a *Dual
// is created for a variable (i.e. the default derivative is 1), then use NewVar.
//
// Additional notes: this is analogous to `unit` or `pure` in Haskell.
// A *Dual is a monadic representation of a dual value.
func New(v values.Value) *Dual {
	// formerly known as dvUnit
	if dv, ok := v.(*Dual); ok {
		return dv
	}
	return constantDV(v)
}

// NewVar creates a new *Dual assuming that the provided value is to be treated as a variable.
//
// Other behaviours from New() is preserved.
func NewVar(v values.Value) *Dual {
	// formerly known as dvUnitVar
	if dv, ok := v.(*Dual); ok {
		return dv
	}
	return variableDV(v)
}

// Bind performs the operation on the inputs. The result is a *Dual that assumes it is a constant value.
//
// (F f) (F a) = F (f a)
func Bind(op Op, inputs ...*Dual) (retVal *Dual, err error) {
	var ret values.Value
	if ret, err = op(idValue(inputs)...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	return New(ret), nil
}

// BindVar performs the operation on the inputs. The result is a *Dual that assumes that it is a variable value.
func BindVar(op Op, inputs ...*Dual) (retVal *Dual, err error) {
	var ret values.Value
	if ret, err = op(idValue(inputs)...); err != nil {
		return nil, errors.Wrap(err, gerrors.OpDoFail)
	}
	return NewVar(ret), nil
}

// Bind0 performs the operation using a preallocated *Dual. The resulting deriv is not set.
func Bind0(op PreallocOp, retVal *Dual, inputs ...*Dual) (*Dual, error) {
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

// Lift transforms a Op into a function that takes the equivalent in *Duals.
func Lift(op Op) func(values ...*Dual) (*Dual, error) {
	return func(inputs ...*Dual) (retVal *Dual, err error) {
		var ret values.Value
		if ret, err = op(idValue(inputs)...); err != nil {
			return nil, errors.Wrap(err, gerrors.OpDoFail)
		}
		return New(ret), nil
	}
}
