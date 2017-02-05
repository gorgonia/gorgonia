package gorgonia

import (
	"github.com/chewxy/gorgonia/tensor"
	tf64 "github.com/chewxy/gorgonia/tensor/f64"

	tf32 "github.com/chewxy/gorgonia/tensor/f32"
)

var (
	/* scalar-tensor float64 and vice versa */

	// arith
	taddf64 = tf64BinOp(tf64.Add)
	tsubf64 = tf64BinOp(tf64.Sub)
	tmulf64 = tf64BinOp(tf64.PointwiseMul)
	tdivf64 = tf64BinOp(tf64.PointwiseDiv)
	tpowf64 = tf64BinOp(tf64.PointwisePow)

	// cmp
	tltf64  = tf64CmpOp(tf64.Lt)
	tgtf64  = tf64CmpOp(tf64.Gt)
	tltef64 = tf64CmpOp(tf64.Lte)
	tgtef64 = tf64CmpOp(tf64.Gte)
	teqf64  = tf64CmpOp(tf64.Eq)
	tnef64  = tf64CmpOp(tf64.Ne)

	/* tf32 */

	taddf32 = tf32BinOp(tf32.Add)
	tsubf32 = tf32BinOp(tf32.Sub)
	tmulf32 = tf32BinOp(tf32.PointwiseMul)
	tdivf32 = tf32BinOp(tf32.PointwiseDiv)
	tpowf32 = tf32BinOp(tf32.PointwisePow)

	// cmp
	tltf32  = tf32CmpOp(tf32.Lt)
	tgtf32  = tf32CmpOp(tf32.Gt)
	tltef32 = tf32CmpOp(tf32.Lte)
	tgtef32 = tf32CmpOp(tf32.Gte)
	teqf32  = tf32CmpOp(tf32.Eq)
	tnef32  = tf32CmpOp(tf32.Ne)
)

type tf32BinOp func(a, b interface{}, opts ...tensor.FuncOpt) (*tf32.Tensor, error)
type tf32CmpOp func(a, b interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error)
type tf64BinOp func(a, b interface{}, opts ...tensor.FuncOpt) (*tf64.Tensor, error)
type tf64CmpOp func(a, b interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error)

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

var tf64BinOps = [maxʘBinaryOpType]*tf64BinOp{
	&taddf64,
	&tsubf64,
	&tmulf64,
	&tdivf64,
	&tpowf64,
	nil, // lt
	nil, // gt
	nil, // lte
	nil, // gte
	nil, // eq
	nil, // ne
}

var tf64CmpOps = [maxʘBinaryOpType]*tf64CmpOp{
	nil, // add
	nil, // sub
	nil, // mul
	nil, // div
	nil, // pow
	&tltf64,
	&tgtf64,
	&tltef64,
	&tgtef64,
	&teqf64,
	&tnef64,
}

var tf32BinOps = [maxʘBinaryOpType]*tf32BinOp{
	&taddf32,
	&tsubf32,
	&tmulf32,
	&tdivf32,
	&tpowf32,
	nil, // lt
	nil, // gt
	nil, // lte
	nil, // gte
	nil, // eq
	nil, // ne
}

var tf32CmpOps = [maxʘBinaryOpType]*tf32CmpOp{
	nil, // add
	nil, // sub
	nil, // mul
	nil, // div
	nil, // pow
	&tltf32,
	&tgtf32,
	&tltef32,
	&tgtef32,
	&teqf32,
	&tnef32,
}
