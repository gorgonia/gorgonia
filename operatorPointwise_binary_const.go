package gorgonia

import "github.com/chewxy/gorgonia/tensor"

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
	teq  = denseCmpOp(tensor.Eq)
	tne  = denseCmpOp(tensor.Ne)
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

var ʘBinOpDiffFns = [maxʘBinaryOpType]func(x, y, z *Node) error{
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
func (b ʘBinaryOperatorType) isCommutative() bool {
	if b >= maxʘBinaryOpType {
		panic("isCommutative() for unsupported BinOp undefined")
	}
	return ʘBinOpCommutative[b]
}

func (b ʘBinaryOperatorType) diffWRT(inputs int) []bool {
	if inputs != 2 {
		panic("binary operator only supports 2 inputs")
	}

	if b.isArith() {
		return []bool{true, true}
	}
	return []bool{false, false}
}

// isArith indicates if the binary operator is an arithmetic type
func (b ʘBinaryOperatorType) isArith() bool {
	switch b {
	case addOpType, subOpType, mulOpType, divOpType, powOpType:
		return true
	default:
		return false
	}
	return false
}

var binOps = [maxʘBinaryOpType]*tf64BinOp{
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

var cmpOps = [maxʘBinaryOpType]*tf64CmpOp{
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
