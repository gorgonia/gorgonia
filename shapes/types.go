package shapes

import (
	"fmt"

	"github.com/chewxy/hm"
)

// types.go allows shapes to be typed

// TypedShape is a type associated with a Shape expression.
type TypedShape struct {
	Expr
	hm.Type
}

func (t TypedShape) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "(%v, %v)", t.Expr, t.Type)
}

func (t *TypedShape) Name() string { return fmt.Sprintf("%v", t) }

func (t *TypedShape) Apply(subs hm.Subs) hm.Substitutable {
	t.Type = t.Type.Apply(subs).(hm.Type)
	return t
}

func (t *TypedShape) FreeTypeVar() hm.TypeVarSet {
	return t.Type.FreeTypeVar()
}

func (t *TypedShape) Normalize(k, v hm.TypeVarSet) (hm.Type, error) {
	ty, err := t.Type.Normalize(k, v)
	if err != nil {
		return nil, err
	}
	t.Type = ty
	return t, nil
}

func (t *TypedShape) Types() hm.Types { return hm.Types{t.Type} }

func (t *TypedShape) Eq(other hm.Type) bool {
	if ot, ok := other.(*TypedShape); ok {
		if !eq(ot.Expr, t.Expr) {
			return false
		}
		return t.Type.Eq(ot.Type)
	}
	return false
}

func (t *TypedShape) String() string { return fmt.Sprintf("%v", t) }

func (t *TypedShape) Clone() interface{} {
	retVal := new(TypedShape)
	retVal.Expr = t.Expr
	retVal.Type = t.Type
	return retVal
}
