package shapes

import "fmt"

type substitution struct {
	Sub Expr
	For Var
}

func (s substitution) Format(f fmt.State, r rune) { fmt.Fprintf(f, "{%v/%v}", s.Sub, s.For) }

type substitutions []substitution

func compose(a, b substitutions) substitutions {
	if len(b) == 0 {
		return a
	}

	retVal := make(substitutions, len(b), len(b)+len(a))
	copy(retVal, b)
	if len(a) == 0 {
		return retVal
	}

	retVal = append(retVal, a...)

	for _, s := range retVal {
		retVal = appendSubs(retVal, substitution{
			For: s.For,
			Sub: s.Sub.apply(a).(Expr),
		})
	}
	return retVal
}

func appendSubs(ss substitutions, s substitution) substitutions {
	for _, s2 := range ss {
		if s2.For == s.For {
			return ss
		}
	}
	ss = append(ss, s)

	return ss
}
