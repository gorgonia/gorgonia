package shapes

import (
	"fmt"

	"github.com/pkg/errors"
)

// OpType represents the type of operation that is being performed
type OpType byte

const (
	// Unary: a → b

	Const OpType = iota // K
	Dims
	Prod
	Sum

	// Binary: a → a → a

	Add
	Sub
	Mul
	Div

	// Cmp: a → a → Bool

	Eq
	Ne
	Lt
	Gt
	Lte
	Gte
)

// String returns the string representation
func (o OpType) String() string {
	switch o {
	case Const:
		return "K"
	case Dims:
		return "D"
	case Prod:
		return "Π"
	case Sum:
		return "Σ"
	case Add:
		return "+"
	case Sub:
		return "-"
	case Mul:
		return "×"
	case Div:
		return "÷"
	case Eq:
		return "="
	case Ne:
		return "≠"
	case Lt:
		return "<"
	case Gt:
		return ">"
	case Lte:
		return "≤"
	case Gte:
		return "≥"
	default:
		return fmt.Sprintf("UNFORMATTED OPTYPE %d", byte(o))
	}
}

// BinOp represents a binary operation.
// It is only an Expr for the purposes of being a value in a shape.
// On the toplevel, BinOp on is meaningless. This is enforced in the `unify` function.
type BinOp struct {
	Op OpType
	A  Expr
	B  Expr
}

func (op BinOp) isSizelike() {}

func (op BinOp) isValid() bool { return op.Op >= Add && op.Op < Eq }

func (op BinOp) resolve() (Size, error) {
	if len(op.A.freevars()) > 0 {
		return 0, errors.Errorf("Cannot resolve BinOp %v - free vars found in A", op)
	}
	if len(op.B.freevars()) > 0 {
		return 0, errors.Errorf("Cannot resolve BinOp %v - free vars found in B", op)
	}

	// Note: the following two checks are unnecessary.  It should always be Size.
	// TODO: get a theorem prover to prove this.

	A, ok := op.A.(Size)
	if !ok {
		return 0, errors.Errorf("Cannot resolve BinOp %v - A is not resolved to a Size", op)
	}

	B, ok := op.B.(Size)
	if !ok {
		return 0, errors.Errorf("Cannot resolve BinOp %v - B is not resolved to a Size", op)
	}

	switch op.Op {
	case Add:
		return A + B, nil
	case Sub:
		return A - B, nil
	case Mul:
		return A * B, nil
	case Div:
		return A / B, nil
	default:
		panic("unreachable")
	}

}

// BinOp implements substitutable

func (op BinOp) apply(ss substitutions) substitutable {
	return BinOp{
		Op: op.Op,
		A:  op.A.apply(ss).(Expr),
		B:  op.B.apply(ss).(Expr),
	}
}

func (op BinOp) freevars() varset {
	retVal := op.A.freevars()
	retVal = append(retVal, op.B.freevars()...)
	return unique(retVal)
}

func (op BinOp) subExprs() []substitutableExpr {
	return []substitutableExpr{op.A.(substitutableExpr), op.B.(substitutableExpr)}
}

// Format formats the BinOp into a nice string.
func (op BinOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v %v %v", op.A, op.Op, op.B) }

// UnaryOp represetns a unary operation on a shape expression.
// Unlike BinaryOp, UnaryOp is an expression.
type UnaryOp struct {
	Op OpType
	A  Expr
}

func (op UnaryOp) isSizelike() {}

func (op UnaryOp) isValid() bool { return op.Op < Add }

func (op UnaryOp) resolve() (Size, error) {
	if !op.isValid() {
		return 0, errors.Errorf("Invalid UnaryOp %v", op)
	}
	if len(op.A.freevars()) > 0 {
		return 0, errors.Errorf("Cannot resolve UnaryOp %v - free vars found in A", op)
	}
	switch A := op.A.(type) {
	case Abstract:
		switch op.Op {
		case Const:
			// ????????? TODO
		case Dims:
			return Size(len(A)), nil
		case Prod:
			retVal := 1
			for _, av := range A {
				switch a := av.(type) {
				case Size:
					retVal *= int(a)
				case Operation:
					v, err := a.resolve()
					if err != nil {
						return 0, errors.Wrapf(err, "Unable to resolve %v.", op)
					}
					retVal *= int(v)
				default:
					return 0, errors.Errorf("Unreachable: a sizelike of %T cannot be Prod'd", av)
				}
			}
			return Size(retVal), nil
		case Sum:
			retVal := 0
			for _, av := range A {
				switch a := av.(type) {
				case Size:
					retVal += int(a)
				case Operation:
					v, err := a.resolve()
					if err != nil {
						return 0, errors.Wrapf(err, "Unable to resolve %v.", op)
					}
					retVal += int(v)
				}
			}
			return Size(retVal), nil
		default:
			panic("unreachable")
		}
	case Shape:
		switch op.Op {
		case Const:
			/// ????? TODO maybe change the signature of resolve()
		case Dims:
			return Size(len(A)), nil
		case Prod:
			retVal := 1
			for i := range A {
				retVal *= A[i]
			}
			return Size(retVal), nil
		case Sum:
			retVal := 0
			for i := range A {
				retVal += A[i]
			}
			return Size(retVal), nil
		default:
			panic("unreachable")
		}
	case Axes:
		// only D is allowed. Error otherwise
		if op.Op != Dims {
			return 0, errors.Errorf("Expected only Dims to work with Axes")
		}
		return Size(len(A)), nil
	case Size:
		switch op.Op {
		case Const:
		// ???? TODO
		case Dims:
			return 0, nil
		case Prod:
			return A, nil
		case Sum:
			return A, nil
		}

	case Axis:
		switch op.Op {
		case Const:
		// ???? TODO
		case Dims:
			return 0, nil
		case Prod:
			return Size(A), nil
		case Sum:
			return Size(A), nil
		}
	default:
		panic("Unreachable")
	}
	panic("Unreachable")
}

// UnaryOp implements substitutable

func (op UnaryOp) apply(ss substitutions) substitutable {
	return UnaryOp{
		Op: op.Op,
		A:  op.A.apply(ss).(Expr),
	}
}
func (op UnaryOp) freevars() varset { return op.A.freevars() }

// UnaryOp is an Expr
func (op UnaryOp) isExpr() {}

// Exprs returns the expression contained within the UnaryOp expression.
func (op UnaryOp) subExprs() []substitutableExpr {
	return []substitutableExpr{op.A.(substitutableExpr)}
}

// Format makes UnaryOp implement fmt.Formatter.
func (op UnaryOp) Format(s fmt.State, r rune) { fmt.Fprintf(s, "%v %v", op.Op, op.A) }
