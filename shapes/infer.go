package shapes

import (
	"fmt"
	"log"
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
			panic("Stuck")
		}
		st = at.SubjectTo
	default:
		panic("Stuck")
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
	log.Printf("st %v", st)
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
	if o, ok := retVal.(Operation); ok {
		if !o.isValid() {
			return nil, errors.Errorf("Failed to infer - final expression %v is an invalid operation", retVal)
		}
		sz, err := o.resolveSize()
		if err != nil {
			return retVal, errors.Wrapf(err, "Cannot resolve final expresison. But it may still be used.")
		}
		return Shape{int(sz)}, nil
	}

	if ce.st.A != nil && ce.st.B != nil {
		log.Printf("st exists")
		st := ce.st.apply(subs).(SubjectTo)
		if ok, err := st.resolveBool(); !ok || err != nil {
			return nil, errors.Errorf("Failed to resolve SubjectTo %v. Error %v", st, err)
		}
		if _, ok := retVal.(Shape); ok {
			// strip the subject to
			return retVal, nil
		}
		return Compound{Expr: retVal, SubjectTo: st}, nil
	}

	return retVal, nil
}

func InferApp(a Expr, b Expr) (Expr, error) {
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
