package shapes

import (
	"fmt"
	"reflect"

	"github.com/pkg/errors"
)

// solver.go implements the constraint solvers
// there are two kinds of constraints to solve: variable constraints and SubjectTo constraints.

// exprConstraint says that A must be equal to B
type exprConstraint struct {
	a, b Expr
}

func (c exprConstraint) apply(ss substitutions) substitutable {
	return exprConstraint{
		a: c.a.apply(ss).(Expr),
		b: c.b.apply(ss).(Expr),
	}
}

func (c exprConstraint) freevars() varset { return exprtup(c).freevars() }

func (c exprConstraint) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "{%v = %v}", c.a, c.b)
}

type constraints []exprConstraint

func (cs constraints) apply(ss substitutions) substitutable {
	if len(ss) == 0 || len(cs) == 0 {
		return cs
	}
	for i := range cs {
		cs[i] = cs[i].apply(ss).(exprConstraint)
	}
	return cs
}

func (cs constraints) freevars() (retVal varset) {
	for i := range cs {
		retVal = append(retVal, cs[i].freevars()...)
	}
	return unique(retVal)
}

type subjectsTo []SubjectTo

func solve(cs constraints, subs substitutions) (newSubs substitutions, err error) {
	switch len(cs) {
	case 0:
		return subs, nil
	default:
		var ss substitutions
		c := cs[0]
		if ss, err = unify(c.a.(substitutableExpr), c.b.(substitutableExpr)); err != nil {
			return nil, err
		}
		newSubs = compose(ss, subs)
		cs2 := cs[1:].apply(newSubs).(constraints)
		return solve(cs2, newSubs)
	}
}

func unify(a, b substitutableExpr) (ss substitutions, err error) {
	switch at := a.(type) {
	case Var:
		return bind(at, b)
	default:
		if eq(a, b) {
			return nil, nil
		}
		if v, ok := b.(Var); ok {
			return bind(v, a)
		}

		aExprs := a.subExprs()
		bExprs := b.subExprs()
		if len(aExprs) == 0 && len(bExprs) == 0 {
			return nil, errors.Errorf("Unification Fail. %v ~ %v cannot proceed", a, b)
		}
		if len(aExprs) != len(bExprs) {
			return nil, errors.Errorf("Unification Fail. %v ~ %v cannot proceed as they do not contain the same amount of sub-expressions. %v has %d subexpressions while %v has %d subexpressions", a, b, a, len(aExprs), b, len(bExprs))
		}
		return unifyMany(aExprs, bExprs)
	}
	panic("TODO")
}

func unifyMany(as, bs []substitutableExpr) (ss substitutions, err error) {
	for i, a := range as {
		b := bs[i]
		if len(ss) > 0 {
			a = a.apply(ss).(substitutableExpr)
			b = b.apply(ss).(substitutableExpr)
		}

		var s2 substitutions
		if s2, err = unify(a, b); err != nil {
			return nil, err
		}

		if ss == nil {
			ss = s2
		} else {
			ss = compose(ss, s2)
		}
	}
	return
}

// tmp solution
func eq(a, b interface{}) bool {
	return reflect.DeepEqual(a, b)
}

func bind(v Var, E substitutable) (substitutions, error) {
	if occurs(v, E) {
		return nil, errors.Errorf("Recursive unification")
	}
	return substitutions{{Sub: E.(Expr), For: v}}, nil
}

func occurs(v Var, in substitutable) bool {
	vs := in.freevars()
	return vs.Contains(v)
}
