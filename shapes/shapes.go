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
		return -1
	case Mul, Div:
		return -1

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
		return op.A.TotalSize()
	case Index:
		return -1
	default:
	}
	panic("Unreachable")
}

type OpType byte

const (
	// Unary
	Const OpType = iota // K
	Index               // []

	// Binary
	Arrow // →
	App   // @
	Add   // +
	Sub   // -
	Mul   // ×
	Div   // ÷
)

/* Example

MatMul:
(a, b) → (b, c) → (a, c)

is represented as:

BinOp{
	Arrow,
	BinOp{
		Arrow,
		tensor.Shape{-1, -2},
		tensor.Shape{-2, -3},
	},
	tensor.Shape{-1, -3},
}


Flatten/Ravel:
(a, b) → (a × b)

is represented as:

BinOp{
	Arrow,
	tensor.Shape{-1, -2},
	BinOp{
		Mul,
		-1, -2,
	}
}

At:

Sum:


*/
