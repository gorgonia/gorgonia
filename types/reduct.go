package types

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/shapes"
)

// Reduct represents the type of a reduction.
type Reduct struct {
	of    hm.Type
	along shapes.Axes
}

func MakeReduct(of hm.Type, along shapes.Axes) Reduct { return Reduct{of, along} }

func (t Reduct) Apply(sub hm.Subs) hm.Substitutable {
	return Reduct{
		of:    t.of.Apply(sub).(hm.Type),
		along: t.along,
	}
}

func (t Reduct) FreeTypeVar() hm.TypeVarSet { return t.of.FreeTypeVar() }

func (t Reduct) Name() string { return "/" }

func (t Reduct) Normalize(k hm.TypeVarSet, v hm.TypeVarSet) (hm.Type, error) {
	dt, err := t.of.Normalize(k, v)
	if err != nil {
		return nil, errors.Wrapf(err, "Normalize %v.of: %v", t, t.of)
	}
	return Reduct{of: dt, along: t.along}, nil
}

func (t Reduct) Types() hm.Types {
	ts := hm.BorrowTypes(1)
	ts[0] = t.of
	return ts
}

func (t Reduct) Eq(other hm.Type) bool {
	switch ot := other.(type) {
	case Reduct:
		return t.of == ot.of && t.along.Eq(ot.along)
	case *Reduct:
		return t.of == ot.of && t.along.Eq(ot.along)
	}
	return false
}

func (t Reduct) Format(f fmt.State, verb rune) { fmt.Fprintf(f, "/[%d] %v", t.along, t.of) }

func (t Reduct) String() string { return fmt.Sprintf("%v", t) }

func (t Reduct) ResolveDepends() hm.Type {
	var of hm.Type
	var dims int
	switch o := t.of.(type) {
	case TensorType:
		of = o.Of
		dims = o.Dims
	case *TensorType:
		of = o.Of
		dims = o.Dims
	default:
		return t.of
	}

	// if no along, then it's a reduction on all axes
	// so we return the scalar
	if len(t.along) == 0 {
		return of
	}

	dims -= len(t.along)
	if dims < 0 {
		panic("Something wrong has happened")
	}
	return TensorType{Dims: dims, Of: of}
}
