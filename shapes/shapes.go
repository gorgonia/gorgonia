package shapes

import "gorgonia.org/tensor"

var (
	_ Expr = tensor.Shape{}
)

type Expr interface {
	TotalSize() int
}

type Var int

func (v Var) TotalSize() int { return -1 }
func (v Var) isValid() bool  { return v < 0 }

type Axis int

func (a Axis) TotalSize() int { return 1 }
func (a Axis) isValid() bool  { return a >= 0 }

type BinOp struct {
	A  Expr
	B  Expr
	Op OpType
}

func (op BinOp) TotalSize() int {
	switch op.Op {
	case Arrow:
		return 0
	case Add, Sub:
	case Mul, Div:

	}
	panic("Unreachable")
}

// UnaryOp represetns a unary operation on a shape expression
type UnaryOp struct {
	Op OpType
	A  Expr
}

func (op UnaryOp) TotalSize() int {
	switch op.Op {
	case Const:
	case Index:
	default:
	}
	panic("Unreachable")
}

type OpType byte

const (
	// Unary
	Const OpType = iota
	Index

	// Binary
	Arrow
	Add
	Sub
	Mul
	Div
)

// (a, b) -> (b, a)

// (a, b) -> (a + b)

// (a, b) -> (a * b, 1)
