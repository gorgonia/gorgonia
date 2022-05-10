package types

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// Dependent is a tensor-type like "result" type that is dependent on the previous type.
type Dependent struct {
	dim   hm.Type // the Dim of the result is dependent on this given type
	dtype hm.Type // the Dtype of the result is dependent of this given type.
}

// MakeDependent creates a new Dependent type. `dim` refers to the type of which the final dims will be reliant on. `dt` refrs to the type of which the final dt will be reliant on.
func MakeDependent(dim hm.Type, dt hm.Type) Dependent { return Dependent{dim: dim, dtype: dt} }

// Apply applies a list of substitutes and returns a copy of itself.
func (t Dependent) Apply(sub hm.Subs) hm.Substitutable {
	dim := t.dim.Apply(sub).(hm.Type)
	dt := t.dtype.Apply(sub).(hm.Type)
	return Dependent{dim: dim, dtype: dt}
}

// FreeTypeVar returns the set of free variables.
func (t Dependent) FreeTypeVar() hm.TypeVarSet {
	set := t.dim.FreeTypeVar()
	return set.Union(t.dtype.FreeTypeVar())
}

// Name returns the name of the dependent type.
func (t Dependent) Name() string { return "⫪" } // double down tack = "depends on". double up tack = "does not depend on"

// Normalize normalizes the type variables in the dependent type, and returns a new copy of the type with the type variables normalized.
func (t Dependent) Normalize(k hm.TypeVarSet, v hm.TypeVarSet) (hm.Type, error) {
	var err error
	var dim, dt hm.Type
	if dim, err = t.dim.Normalize(k, v); err != nil {
		return nil, errors.Wrapf(err, "Normalize %v.Dim", t)
	}
	if dt, err = t.dtype.Normalize(k, v); err != nil {
		return nil, errors.Wrapf(err, "Normalize %v.Dtype", t)
	}

	return Dependent{dim: dim, dtype: dt}, nil
}

// Types returns the inner types.
func (t Dependent) Types() hm.Types {
	ts := hm.BorrowTypes(2)
	ts[0], ts[1] = t.dim, t.dtype
	return ts
}

// Eq returns true when either a Dependent or a *Dependent is equal.
// Anything else returns false.
func (t Dependent) Eq(other hm.Type) bool {
	switch ot := other.(type) {
	case Dependent:
		return t.dim.Eq(ot.dim) && t.dtype.Eq(ot.dtype)
	case *Dependent:
		return t.dim.Eq(ot.dim) && t.dtype.Eq(ot.dtype)
	}
	return false
}

// Format implements fmt.Formatter.
func (t Dependent) Format(f fmt.State, verb rune) { fmt.Fprintf(f, "⫪[%v %v]", t.dim, t.dtype) }

// String implements fmt.Stringer
func (t Dependent) String() string { return fmt.Sprintf("%v", t) }

func (t Dependent) ResolveDepends() hm.Type {
	var dim int
	switch d := t.dim.(type) {
	case TensorType:
		dim = d.Dims
	case *TensorType:
		dim = d.Dims
	}

	var of hm.Type
	switch o := t.dtype.(type) {
	case TensorType:
		of = o.Of
	case *TensorType:
		of = o.Of
	default:
		of = t.dtype
	}

	if dim == 0 {
		return of
	}

	return TensorType{Dims: dim, Of: of}
}
