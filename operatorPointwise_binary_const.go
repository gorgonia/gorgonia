package gorgonia

import "gorgonia.org/tensor"

var (
	/* scalar-tensor float64 and vice versa */

	// arith
	tadd = denseBinOp(tensor.Add)
	tsub = denseBinOp(tensor.Sub)
	tmul = denseBinOp(tensor.Mul)
	tdiv = denseBinOp(tensor.Div)
	tpow = denseBinOp(tensor.Pow)

	// cmp
	tlt  = denseCmpOp(tensor.Lt)
	tgt  = denseCmpOp(tensor.Gt)
	tlte = denseCmpOp(tensor.Lte)
	tgte = denseCmpOp(tensor.Gte)
	teq  = denseCmpOp(tensor.ElEq)
	tne  = denseCmpOp(tensor.ElNe)
)

type denseBinOp func(a, b interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error)
type denseCmpOp func(a, b interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error)

type ʘBinaryOperatorType byte

const (
	// arith
	addOpType ʘBinaryOperatorType = iota
	subOpType
	mulOpType
	divOpType
	powOpType

	// cmp
	ltOpType
	gtOpType
	lteOpType
	gteOpType
	eqOpType
	neOpType

	maxʘBinaryOpType // delimits the end of all possible binOpType
)

func (op ʘBinaryOperatorType) String() string {
	return ʘBinOpStrs[op]
}

// ʘBinOpStrs is the string representation for a binOpType
// It should be held constant.
var ʘBinOpStrs = [maxʘBinaryOpType]string{
	// arith ops
	"+",
	"-",
	"⊙",
	"÷",
	"^",

	// cmp ops
	"<",
	">",
	"<=",
	">=",
	"==",
	"!=",
}

// ʘBinOpNames is the string representation for a binOpType
// It should be held constant.
var ʘBinOpNames = [maxʘBinaryOpType]string{
	// arith ops
	"add",
	"sub",
	"mul",
	"div",
	"pow",

	// cmp ops
	"lt",
	"gt",
	"lte",
	"gte",
	"eq",
	"ne",
}

// ʘBinOpCommutative is the array that stores whether a binary operator is commutative
// It should be held constant.
var ʘBinOpCommutative = [maxʘBinaryOpType]bool{
	true, false, true, false, false,
	false, false, false, false, true, true,
}

var ʘBinOpDiffExprs = [maxʘBinaryOpType]func(x, y, z, gradZ *Node) (Nodes, error){
	addDiffExpr, subDiffExpr, hadamardProdDiffExpr, hadamardDivDiffExpr, hadamardPowDiffExpr,
	nondiffBinOpExpr, nondiffBinOpExpr, nondiffBinOpExpr, nondiffBinOpExpr, nondiffBinOpExpr, nondiffBinOpExpr,
}

var ʘBinOpDiffFns = [maxʘBinaryOpType]func(ctx ExecutionContext, x, y, z *Node) error{
	addDiff, subDiff, hadamardProdDiff, hadamardDivDiff, hadamardPowDiff,
	nondiffBinOp, nondiffBinOp, nondiffBinOp, nondiffBinOp, nondiffBinOp, nondiffBinOp,
}

// isCommutative gives info about whether the operator is commutative
// For example:
//		a + b == b + a
// will ALWAYS evaluate to true. The same cannot be said about subtraction:
// 		a - b != b - a
// While a-b *may* be equal to b-a, it is not guaranteed. Therefore subtraction
// is not commutative
func (op ʘBinaryOperatorType) isCommutative() bool {
	if op >= maxʘBinaryOpType {
		panic("isCommutative() for unsupported BinOp undefined")
	}
	return ʘBinOpCommutative[op]
}

func (op ʘBinaryOperatorType) diffWRT(inputs int) []bool {
	if inputs != 2 {
		panic("binary operator only supports 2 inputs")
	}

	if op.isArith() {
		return []bool{true, true}
	}
	return []bool{false, false}
}

// isArith indicates if the binary operator is an arithmetic type
func (op ʘBinaryOperatorType) isArith() bool {
	switch op {
	case addOpType, subOpType, mulOpType, divOpType, powOpType:
		return true
	default:
		return false
	}
}

var binOps = [maxʘBinaryOpType]*denseBinOp{
	&tadd,
	&tsub,
	&tmul,
	&tdiv,
	&tpow,
	nil, // lt
	nil, // gt
	nil, // lte
	nil, // gte
	nil, // eq
	nil, // ne
}

var cmpOps = [maxʘBinaryOpType]*denseCmpOp{
	nil, // add
	nil, // sub
	nil, // mul
	nil, // div
	nil, // pow
	&tlt,
	&tgt,
	&tlte,
	&tgte,
	&teq,
	&tne,
}
