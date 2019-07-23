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

type UnaryOp struct {
	Op OpType
	A  Expr
}

type OpType byte

const (
	Const OpType = iota
	Index

	Add
	Sub
	Mul
	Div
)
