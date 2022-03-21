package types

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/shapes"
)

// Sliced represents the type of a tensor which has been sliced.
type Sliced struct {
	Of    hm.Type
	Along shapes.Slices
}

func (t *Sliced) Apply(subs hm.Subs) hm.Substitutable {
	of := t.Of.Apply(subs).(hm.Type)
	t2 := &Sliced{Of: of, Along: t.Along}
	return t2.Canonical()
}

func (t *Sliced) sliceOne(curDim int, sl shapes.Slice) (newDim int) {
	var selection int
	if sl == nil {
		selection = -1
	} else {
		selection = sl.End() - sl.Start()
	}
	if selection == 1 {
		return curDim - 1
	}
	return curDim
}

func (t *Sliced) FreeTypeVar() hm.TypeVarSet { return t.Of.FreeTypeVar() }

func (t *Sliced) Name() string { return "Sliced" }

func (t *Sliced) Normalize(k hm.TypeVarSet, v hm.TypeVarSet) (hm.Type, error) {
	var err error
	if t.Of, err = t.Of.Normalize(k, v); err != nil {
		return nil, err
	}

	return t, nil
}

func (t *Sliced) Types() hm.Types {
	ts := hm.BorrowTypes(1)
	ts[0] = t.Of
	return ts
}

func (t *Sliced) Eq(other hm.Type) bool {
	switch ot := other.(type) {
	case *Sliced:
		if !ot.Of.Eq(t.Of) {
			return false
		}
		if len(ot.Along) != len(t.Along) {
			return false
		}
		for i, s1 := range t.Along {
			s2 := ot.Along[i]
			if s1.Start() != s2.Start() || s1.End() != s2.End() || s1.Step() != s2.Step() {
				return false
			}
		}
		return true
	default:
		return false
	}
	panic("Unreachable")
}

func (t *Sliced) Format(b fmt.State, verb rune) { fmt.Fprintf(b, "%v%v", t.Of, t.Along) }

func (t *Sliced) String() string { return fmt.Sprintf("%v", t) }

func (t *Sliced) Canonical() hm.Type {
	switch o := t.Of.(type) {
	case TensorType:
		d := o.Dims
		for _, sl := range t.Along {
			d = t.sliceOne(d, sl)
		}
		switch {
		case d < 0:
			panic("Impossible situation. Please file a bug on https://github.com/gorgonia/gorgonia")
		case d == 0:
			return o.Of // scalar type
		default:
			return TensorType{Of: o.Of, Dims: d}
		}
	default:
		return t
	}
}
