package shapes

import (
	"fmt"

	"github.com/pkg/errors"
)

// compound.go describes compound terms

// SubjectTo describes a constraint
type SubjectTo struct {
	OpType
	A, B Operation
}

func (s SubjectTo) Format(st fmt.State, r rune) {
	fmt.Fprintf(st, "(%v %v %v)", s.A, s.OpType, s.B)
}

// SubjectTo implements substitutable

func (s SubjectTo) apply(ss substitutions) substitutable {
	return SubjectTo{
		OpType: s.OpType,
		A:      s.A.apply(ss).(Operation),
		B:      s.B.apply(ss).(Operation),
	}
}

func (s SubjectTo) freevars() varset { return append(s.A.freevars(), s.B.freevars()...) }

func (s SubjectTo) subExprs() []substitutableExpr { return []substitutableExpr{s.A, s.B} }

// subjectTo is also an Operation

func (s SubjectTo) isValid() bool { return s.OpType >= Eq && s.A.isValid() && s.B.isValid() }

func (s SubjectTo) resolveSize() (Size, error) {
	return 0, errors.Errorf("SubjectTo does not resolve to Size.")
}

func (s SubjectTo) resolveBool() (bool, error) {
	switch s.OpType {
	case And, Or:
		A, err := s.A.(boolOp).resolveBool()
		if err != nil {
			return false, errors.Wrapf(err, "Failed to resolve operand A of SubjectTo %v into a bool", s)
		}
		B, err := s.B.(boolOp).resolveBool()
		if err != nil {
			return false, errors.Wrapf(err, "Failed to resolve operand A of SubjectTo %v into a bool", s)
		}
		switch s.OpType {
		case And:
			return A && B, nil
		case Or:
			return A || B, nil
		}
		panic("Unreachable")
	default:
		op := BinOp{s.OpType, s.A.(Expr), s.B.(Expr)}
		return op.resolveBool()
	}

}

type Compound struct {
	Expr
	SubjectTo
}

func (c Compound) Format(s fmt.State, r rune) {
	fmt.Fprintf(s, "{ %v | %v }", c.Expr, c.SubjectTo)
}
func (c Compound) apply(ss substitutions) substitutable {
	return Compound{
		Expr:      c.Expr.apply(ss).(Expr),
		SubjectTo: c.SubjectTo,
	}
}
func (c Compound) freevars() varset {
	retVal := c.Expr.freevars()
	retVal = append(retVal, c.SubjectTo.freevars()...)
	return retVal
}
