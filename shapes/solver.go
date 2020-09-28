package shapes

import "reflect"

// solver.go implements the constraint solvers
// there are two kinds of constraints to solve: variable constraints and SubjectTo constraints.

func solve(cs constraints, subs substitutions) (newSubs substitutions, err error) {
	switch len(cs) {
	case 0:
		return subs, nil
	default:
		var ss substitutions
		c := cs[0]
		if ss, err = unify(c.a, c.b); err != nil {
			return err
		}
		newSubs = compose(ss, subs)
		cs2 := cs[1:].apply(newSubs).(constraints)
		return solve(cs2, newSubs)
	}
}

func unify(a, b Expr) (ss substitutions, err error) {
	switch at := a.(type) {
	case Var:
		return substitutions{substitution{Sub: b, For: a}}
	default:
		if eq(a, b) {
			return nil, nil
		}
	}
}

// tmp solution
func eq(a, b Expr) bool {
	return reflect.DeepEqual(a, b)
}
