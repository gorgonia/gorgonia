package shapes

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var lexCases = map[string][]tok{
	"()":         []tok{{parenL, '(', 0}, {parenR, ')', 1}},
	"(a,)":       []tok{{parenL, '(', 0}, {letter, 'a', 1}, {parenR, ')', 3}},
	"(1, 2, 34)": []tok{{parenL, '(', 0}, {digit, 1, 1}, {digit, 2, 4}, {digit, 34, 8}, {parenR, ')', 9}},
	"() -> ()":   []tok{{parenL, '(', 0}, {parenR, ')', 1}, {arrow, '→', 4}, {parenL, '(', 6}, {parenR, ')', 7}},
	"() → ()":    []tok{{parenL, '(', 0}, {parenR, ')', 1}, {arrow, '→', 3}, {parenL, '(', 5}, {parenR, ')', 6}},

	// unop
	"P a": []tok{{unop, 'Π', 0}, {letter, 'a', 2}},
	"S a": []tok{{unop, 'Σ', 0}, {letter, 'a', 2}},

	// binop, cmpop and logop
	"a + 1":  []tok{{letter, 'a', 0}, {binop, '+', 2}, {digit, 1, 4}},
	"a - 1":  []tok{{letter, 'a', 0}, {binop, '-', 2}, {digit, 1, 4}},
	"a = 2":  []tok{{letter, 'a', 0}, {cmpop, '=', 2}, {digit, 2, 4}},
	"a != 2": []tok{{letter, 'a', 0}, {cmpop, '≠', 3}, {digit, 2, 5}},
	"a > 1":  []tok{{letter, 'a', 0}, {cmpop, '>', 2}, {digit, 1, 4}},
	"a >= 1": []tok{{letter, 'a', 0}, {cmpop, '≥', 3}, {digit, 1, 5}},
	"a <= 1": []tok{{letter, 'a', 0}, {cmpop, '≤', 3}, {digit, 1, 5}},
	"a ≥ 1":  []tok{{letter, 'a', 0}, {cmpop, '≥', 2}, {digit, 1, 4}},
	"a ∧ 1":  []tok{{letter, 'a', 0}, {logop, '∧', 2}, {digit, 1, 4}},
	"a && 1": []tok{{letter, 'a', 0}, {logop, '∧', 3}, {digit, 1, 5}},
	"a || 1": []tok{{letter, 'a', 0}, {logop, '∨', 3}, {digit, 1, 5}},

	// constructions
	"{(a) -> () | (a > 2)}": []tok{
		{braceL, '{', 0},
		{parenL, '(', 1},
		{letter, 'a', 2},
		{parenR, ')', 3},
		{arrow, '→', 6},
		{parenL, '(', 8},
		{parenR, ')', 9},
		{pipe, '|', 11},
		{parenL, '(', 13},
		{letter, 'a', 14},
		{cmpop, '>', 16},
		{digit, 2, 18},
		{parenR, ')', 19},
		{braceR, '}', 20},
	},
	"[0:2:1]": []tok{{brackL, '[', 0}, {digit, 0, 1}, {digit, 2, 3}, {digit, 1, 5}, {brackR, ']', 6}},

	// dubious API design wise
	"& a": []tok{{letter, 'a', 2}}, // note that the singular '&' is ignored.
}

func TestLex(t *testing.T) {
	for k, v := range lexCases {
		toks, err := lex(k)
		if err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, v, toks)
	}
}

var parseCases = map[string]Expr{

	"()":           Shape{},
	"(1,)":         Shape{1},
	"(1,2,3,2325)": Shape{1, 2, 3, 2325},
	"(1, a, 2)":    Abstract{Size(1), Var('a'), Size(2)},
	"(a,b,c) → (a*b, b*c)": Arrow{
		Abstract{Var('a'), Var('b'), Var('c')},
		Abstract{
			BinOp{Mul, Var('a'), Var('b')},
			BinOp{Mul, Var('b'), Var('c')},
		},
	},

	// Transpose:
	"{ a → X[0 1 3 2] → Tr X[0 1 3 2] a | (D X[0 1 3 2] = D a) }": Compound{
		Expr: Arrow{
			Var('a'),
			Arrow{
				Axes{0, 1, 3, 2},
				TransposeOf{
					Axes{0, 1, 3, 2},
					Var('a'),
				},
			},
		},
		SubjectTo: SubjectTo{
			Eq,
			UnaryOp{Dims, Axes{0, 1, 3, 2}},
			UnaryOp{Dims, Var('a')},
		},
	},

	// Indexing
	"a → b -> ()": Arrow{
		Var('a'),
		Arrow{Var('b'), Shape{}},
	},

	// Indexing (constrained)
	"{ a → b → () | ((D a = D b) ∧ (∀ b < ∀ a)) }": Compound{
		Expr: Arrow{
			Var('a'),
			Arrow{
				Var('b'),
				Shape{},
			},
		},
		SubjectTo: SubjectTo{
			And,
			SubjectTo{
				Eq,
				UnaryOp{Dims, Var('a')},
				UnaryOp{Dims, Var('b')},
			},
			SubjectTo{
				Lt,
				UnaryOp{ForAll, Var('b')},
				UnaryOp{ForAll, Var('a')},
			},
		},
	},

	// Slicing
	"{ a → [0:2] → a[0:2] | (a[0] ≥ 2) }": Compound{
		Expr: Arrow{
			Var('a'),
			Arrow{
				Sli{0, 2, 1},
				SliceOf{
					Sli{0, 2, 1},
					Var('a'),
				},
			},
		},
		SubjectTo: SubjectTo{
			OpType: Gte,
			A:      IndexOf{I: 0, A: Var('a')},
			B:      Size(2),
		},
	},

	// Reshaping
	"{ a → b → b | (Π a = Π b) }": Compound{
		Arrow{
			Var('a'),
			Arrow{
				Var('b'),
				Var('b'),
			},
		},
		SubjectTo{
			Eq,
			UnaryOp{Prod, Var('a')},
			UnaryOp{Prod, Var('b')},
		},
	},

	// Columnwise Sum Matrix
	"{ a → b | (D b = D a - 1) }": Compound{
		Arrow{
			Var('a'),
			Var('b'),
		},
		SubjectTo{
			Eq,
			UnaryOp{Dims, Var('b')},
			BinOp{
				Sub,
				UnaryOp{Dims, Var('a')},
				Size(1),
			},
		},
	},
}

/*
var knownFail = map[string]Expr{
	"(a,b,c) → (a*b+c, a*b+c)": Arrow{
		Abstract{Var('a'), Var('b'), Var('c')},
		Abstract{
			BinOp{Add, BinOp{Mul, Var('a'), Var('b')}, Var('c')},
			BinOp{Add, BinOp{Mul, Var('a'), Var('b')}, Var('c')},
		},
	},
}
*/

func TestParse(t *testing.T) {
	for k, v := range parseCases {
		expr, err := Parse(k)
		if err != nil {
			t.Fatal(err)
		}
		assert.Equal(t, v, expr, "Failed to parse %q", k)
	}
}
