package shapes

import (
	"reflect"
	"testing"
)

var unifyTests = []struct {
	a, b    substitutableExpr
	correct substitutions
	willerr bool
}{
	{
		// unify (2, 3) with (a, b)
		Shape{2, 3}, Abstract{Var('a'), Var('b')},
		substitutions{substitution{Sub: Size(3), For: Var('b')}, substitution{Sub: Size(2), For: Var('a')}},
		false,
	},

	{
		// unify (a, b) with (2, 3)
		Abstract{Var('a'), Var('b')}, Shape{2, 3},
		substitutions{substitution{Sub: Size(3), For: Var('b')}, substitution{Sub: Size(2), For: Var('a')}},
		false,
	},

	{
		// unify (a, b) with (c, d)
		Abstract{Var('a'), Var('b')}, Abstract{Var('c'), Var('d')},
		substitutions{substitution{Sub: Var('d'), For: Var('b')}, substitution{Sub: Var('c'), For: Var('a')}},
		false,
	},

	{
		// unify (a, b) with (b, c)
		// This is usually a degenerate case.
		// The substitutions need to be applied backwards. But it's impossible to actually tell if this is an error.
		// The result is also typical of one where functions are applied.
		Abstract{Var('a'), Var('b')}, Abstract{Var('b'), Var('c')},
		substitutions{substitution{Sub: Var('c'), For: Var('b')}, substitution{Sub: Var('b'), For: Var('a')}},
		false,
	},

	{
		// unify a with (2,3)
		Var('a'), Shape{2, 3},
		substitutions{substitution{Sub: Shape{2, 3}, For: Var('a')}},
		false,
	},
}

func TestUnify(t *testing.T) {
	for _, ut := range unifyTests {
		subs, err := unify(ut.a, ut.b)
		if err != nil && !ut.willerr {
			t.Errorf("Error (%v ~ %v): %v", ut.a, ut.b, err)
		} else if err == nil && ut.willerr {
			t.Errorf("Expected error when unifying %v ~ %v", ut.a, ut.b)
		}
		if !reflect.DeepEqual(subs, ut.correct) {
			t.Errorf("Error (%v ~ %v): Expected %v, Got %v instead", ut.a, ut.b, ut.correct, subs)
		}
	}
}

var solveTests = []struct {
	cs      constraints
	subs    substitutions
	correct substitutions
	willerr bool
}{
	{constraints{{a: Arrow{Var('a'), Var('b')}, b: Arrow{Var('b'), Var('c')}}},
		substitutions{},
		substitutions{{Var('c'), Var('b')}, {Var('b'), Var('a')}},
		false},
}

func TestSolve(t *testing.T) {
	for _, st := range solveTests {
		subs, err := solve(st.cs, st.subs)
		if err != nil && !st.willerr {
			t.Errorf("Solving %v with %v. Error: %v", st.cs, st.subs, err)
		} else if err == nil && st.willerr {
			t.Errorf("Solving %v wiht %v. Expected an error", st.cs, st.subs)
		}
		if !reflect.DeepEqual(subs, st.correct) {
			t.Errorf("Error while solving %v with %v. Expected %v. Got %v instead", st.cs, st.subs, st.correct, subs)
		}
	}
}
