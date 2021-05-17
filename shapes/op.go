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
	ForAll // special use

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

	// Logic: bool → bool → bool

	And
	Or
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
	case ForAll:
		return "∀"
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
	case And:
		return "∧"
	case Or:
		return "∨"
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

func (op BinOp) resolveSize() (Size, error) {
	if err := op.nofreeevars(); err != nil {
		return 0, errors.Wrapf(err, "Cannot resolve BinOp %v to Size.", op)
	}
	A, B, err := op.resolveAB()
	if err != nil {
		return 0, errors.Wrapf(err, "Cannot resolve BinOp %v to Size.", op)
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
		return 0, errors.Errorf("Unable to resolve op %v into a Size", op)
	}
	panic("unreachable")

}

// resolveBool is only ever used by SubjectTo's resolveBool.
func (op BinOp) resolveBool() (bool, error) {
	if err := op.nofreeevars(); err != nil {
		return false, errors.Wrapf(err, "Cannot resolve BinOp %v to bool.", op)
	}

	A, B, err := op.resolveAB()
	if err != nil {
		// check to see if A and B are ForAlls
		var auo, buo UnaryOp
		var A, B intslike
		var al, bl []int
		var ok bool

		if auo, ok = op.A.(UnaryOp); !ok || auo.Op != ForAll {
			goto fail
		}

		if buo, ok = op.B.(UnaryOp); !ok || buo.Op != ForAll {
			goto fail
		}

		if A, ok = auo.A.(intslike); !ok {
			goto fail
		}
		if B, ok = buo.A.(intslike); !ok {
			goto fail
		}

		al, bl = A.AsInts(), B.AsInts()
		if len(al) != len(bl) {
			return false, nil // will always be false because you cannot compare slices of unequal length. Does this mean it should have an error message? Unsure
		}
		bl = bl[:len(al)]
		for i, a := range al {
			b := bl[i]
			ok, err := op.do(Size(a), Size(b))
			if err != nil {
				return false, errors.Wrapf(err, "Cannot resolve %v %v %v (%dth index in %v)", a, op.Op, b, i, op)
			}
			if !ok {
				return false, nil
			}
		}

		return true, nil

	fail:
		return false, errors.Wrapf(err, "Cannot resolve BinOp %v to bool.", op)
	}
	return op.do(A, B)

}

func (op BinOp) do(A, B Size) (bool, error) {
	switch op.Op {
	case Eq:
		return A == B, nil
	case Ne:
		return A != B, nil
	case Lt:
		return A < B, nil
	case Gt:
		return A > B, nil
	case Lte:
		return A <= B, nil
	case Gte:
		return A >= B, nil
	default:
		return false, errors.Errorf("Cannot resolve BinOp %v to bool.", op)

	}
}

func (op BinOp) nofreeevars() error {
	if len(op.A.freevars()) > 0 {
		return errors.Errorf("Cannot resolve BinOp %v - free vars found in A", op)
	}
	if len(op.B.freevars()) > 0 {
		return errors.Errorf("Cannot resolve BinOp %v - free vars found in B", op)
	}
	return nil
}

func (op BinOp) resolveAB() (A, B Size, err error) {
	switch a := op.A.(type) {
	case Size:
		A = a
	case sizeOp:
		var err error
		if A, err = a.resolveSize(); err != nil {
			return A, B, errors.Wrapf(err, "BinOp %v - A (%v) does not resolve to a Size", op, op.A)
		}
	default:
		return 0, 0, errors.Errorf("Cannot resolve BinOp %v - A (%v) is not resolved to a Size", op, op.A)
	}

	switch b := op.B.(type) {
	case Size:
		B = b
	case sizeOp:
		var err error
		if B, err = b.resolveSize(); err != nil {
			return A, B, errors.Wrapf(err, "BinOp %v - B (%v) does not resolve to a Size", op, op.B)
		}
	default:
		return 0, 0, errors.Errorf("Cannot resolve BinOp %v - B is not resolved to a Size", op)
	}
	return
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

func (op UnaryOp) resolveSize() (Size, error) {
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
				case sizeOp:
					v, err := a.resolveSize()
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
				case sizeOp:
					v, err := a.resolveSize()
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
			return 0, errors.Errorf(unaryOpResolveErr, op)
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
			return 0, errors.Errorf(unaryOpResolveErr, op)
		}
		return Size(len(A)), nil
	case Sizes:
		// only D is allowed. Error otherwise
		if op.Op != Dims {
			return 0, errors.Errorf(unaryOpResolveErr, op)
		}
		return Size(len(A)), nil
	case Size:
		switch op.Op {
		case Const:
			return 0, errors.Errorf(unaryOpResolveErr, op)
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
			return 0, errors.Errorf(unaryOpResolveErr, op)
		case Dims:
			return 0, nil
		case Prod:
			return Size(A), nil
		case Sum:
			return Size(A), nil
		}
	default:
		return 0, errors.Errorf(unaryOpResolveErr, op)
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
