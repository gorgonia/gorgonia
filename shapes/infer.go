package shapes

import (
	"fmt"
	"sort"

	"github.com/pkg/errors"
)

// ConstraintExpr is a tuple of a list of constraints and an expression.
type ConstraintsExpr struct {
	cs constraints
	e  Expr
}

func (ce ConstraintsExpr) Format(f fmt.State, r rune) {
	fmt.Fprintf(f, "%v | %v", ce.cs, ce.e)
}

// App applys an expression to a function/Arrow expression.
// This function will aggressively perform alpha renaming on the expression.
//
// Example. Given an application of the following:
// 	((a,b) → (b, c) → (a, c)) @ (2, a)
//
// The variables in the latter will be aggressively renamed, to become:
//	((a,b) → (b, c) → (a, c)) @ (2, d)
//
// Normally this wouldn't be a concern, as you would be passing in concrete shapes, something like:
//	((a,b) → (b, c) → (a, c)) @ (2, 3)
// which will then yield:
// 	(3, c) → (2, c)
func App(a Arrow, b Expr) ConstraintsExpr {
	fv := a.freevars()

	// rename all the free variables in b
	b = alpha(fv, b)

	// add those new free variables to the set of free variables
	fv = append(fv, b.freevars()...)
	fv = unique(fv)

	// get a fresh variable given the set already used
	fr := fresh(fv)
	cs := constraints{{a, Arrow{b, fr}}}
	return ConstraintsExpr{cs, fr}
}

func Infer(ce ConstraintsExpr) (Expr, error) {
	if ce.e == nil {
		return nil, errors.Errorf("No expression found in ConstraintExpr %v", ce)
	}

	subs, err := solve(ce.cs, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to solve %v", ce)
	}

	return ce.e.apply(subs).(Expr), nil
}

func InferApp(ar Expr, b Expr) (Expr, error) {
	a, ok := ar.(Arrow)
	if !ok {
		return nil, errors.Errorf("InferApp can only work if `ar` is an `Arrow` type. Got %T instead", ar)
	}

	return Infer(App(a, b))
}

func fresh(set varset) Var {
	sort.Sort(set)
	if len(set) == 0 {
		return 'a'
	}
	return set[len(set)-1] + 1
}

func alpha(set varset, a Expr) Expr {
	return a // TODO
}
