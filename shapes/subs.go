package shapes

type substitutable interface {
	apply(substitutions) substitutable
	freevars() varset // set of free variables
}

type substitution struct {
	Sub Expr
	For Var
}

type substitutions []substitution

func compose(a, b substitutions) substitutions {
	if len(b) == 0 {
		return a
	}

	retVal := make(substitutions, 0, len(b)+2*len(a))
	copy(retVal, b)
	if len(a) == 0 {
		return retVal
	}

	retVal = append(retVal, a...)

	for _, s := range retVal {
		retVal = append(retVal, substitution{
			For: s.For,
			Sub: s.Sub.apply(a).(Expr),
		})
	}
	return retVal
}
