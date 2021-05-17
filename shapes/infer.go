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
	st SubjectTo
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
func App(ar Expr, b Expr) ConstraintsExpr {
	var a Arrow
	var st SubjectTo
	switch at := ar.(type) {
	case Arrow:
		a = at
	case Compound:
		var ok bool
		if a, ok = at.Expr.(Arrow); !ok {
			panic(fmt.Sprintf("Unhandled type at.Expr %v of  %T", at.Expr, at.Expr))
		}
		st = at.SubjectTo
	default:
		panic(fmt.Sprintf("Unhandled type ar %v of  %T", ar, ar))
	}

	fv := a.freevars()

	// rename all the free variables in b
	b = alpha(fv, b)

	// add those new free variables to the set of free variables
	fv = append(fv, b.freevars()...)
	fv = unique(fv)

	// get a fresh variable given the set already used
	fr := fresh(fv)
	cs := constraints{{a, Arrow{b, fr}}}
	return ConstraintsExpr{cs, fr, st}
}

func Infer(ce ConstraintsExpr) (Expr, error) {
	if ce.e == nil {
		return nil, errors.Errorf("No expression found in ConstraintExpr %v", ce)
	}

	subs, err := solve(ce.cs, nil)
	if err != nil {
		return nil, errors.Wrapf(err, "Failed to solve %v", ce)
	}

	retVal := ce.e.apply(subs).(Expr)
	if retVal, err = recursiveResolve(retVal); err != nil {
		return retVal, err
	}

	if ce.st.A != nil && ce.st.B != nil {
		st := ce.st.apply(subs).(SubjectTo)
		if len(st.freevars()) > 0 {
			// don't try to resolve the st yet.
			return Compound{Expr: retVal, SubjectTo: st}, nil
		}

		ok, err := st.resolveBool()
		if err != nil {
			return nil, errors.Errorf("Failed to resolve SubjectTo %v. Error %v", st, err)
		}
		if !ok {
			return nil, errors.Errorf("SubjectTo %v resolved to false. Cannot continue", st)
		}
		return retVal, nil
	}

	return retVal, nil
}

func InferApp(a Expr, others ...Expr) (retVal Expr, err error) {
	if len(others) == 0 {
		// error
	}
	fst := a
	for _, e := range others {
		if fst, err = Infer(App(fst, e)); err != nil {
			return nil, err
		}

	}
	return fst, nil
}

func ToShape(a Expr) (Shape, error) {
	switch at := a.(type) {
	case Shape:
		return at, nil
	case Abstract:
		sh, ok := at.ToShape()
		if !ok {
			return nil, errors.Errorf("Unable to concretize %v of %T", a, a)
		}
		return sh, nil
	default:
		return nil, errors.Errorf("Unable to concretize %v of %T", a, a)
	}
}

func fresh(set varset) Var {
	sort.Sort(set)
	if len(set) == 0 {
		return 'a'
	}
	return set[len(set)-1] + 1
}

func alpha(set varset, a Expr) Expr {
	fv := a.freevars()
	var subs substitutions
	for _, v := range fv {
		if set.Contains(v) {
			fr := fresh(set)
			set = append(set, fr)
			subs = append(subs, substitution{Sub: fr, For: v})
		}
	}
	a2 := a.apply(subs).(Expr)
	return a2
}

func recursiveResolve(a Expr) (Expr, error) {
	switch at := a.(type) {
	case sizeOp:
		if !at.isValid() {
			return nil, errors.Errorf("Expression %v is not a valid Operation", at)
		}
		sz, err := at.resolveSize()
		if err != nil {
			return a, errors.Wrapf(err, "Cannot resolve final expresison. But it may still be used.")
		}
		return Shape{int(sz)}, nil
	case resolver:
		retVal, err := at.resolve()
		if err != nil {
			return nil, errors.Wrapf(err, "Failed to recursively resolve %v", at)
		}
		return recursiveResolve(retVal)
	default:
		// nothing else can be resolved. return the identity
		return a, nil
	}
}
