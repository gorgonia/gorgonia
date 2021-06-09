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
