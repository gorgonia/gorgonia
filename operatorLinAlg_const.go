package gorgonia

import "github.com/chewxy/hm"

// āBinOpStrs is the string representation for binLAOperator
// It should be held constant
var āBinOpStrs = [maxĀBinaryOperator]string{
	"×",
	"×",
	"⋅",
	"⊗",
	// "×××",
}

var āBinOpDiffExprs = [maxĀBinaryOperator]func(tA, tB bool, x, y, z, grad *Node) (Nodes, error){
	matMulDiffExpr,
	matVecMulDiffExpr,
	vecDotDiffExpr,
	outerProdDiffExpr,
}

var āBinOpDiffs = [maxĀBinaryOperator]func(tA, tB bool, x, y, z *Node) error{
	matMulDiff,
	matVecMulDiff,
	vecDotDiff,
	outerProdDiff,
}

var āBinOpTypes = [maxĀBinaryOperator]func() hm.Type{
	matMulType,
	matVecMulType,
	vecDotType,
	outerProdType,
}

/* TYPES FOR LINALG BINARY OP*/

// matVecMulOp is a function with this type:
//		matVecMulOp :: (Float a) ⇒ Vector a → Matrix a → Vector a
//
// For the moment only floats are allowed
func matVecMulType() hm.Type {
	a := hm.TypeVariable('a')
	v := newTensorType(1, a)
	m := newTensorType(2, a)

	return hm.NewFnType(m, v, v)
}

// matMulOp is a function with this type:
//		matMulOp :: (Float a) ⇒ Matrix a → Matrix a → Matrix a
//
// For the moment only floats are allowed
func matMulType() hm.Type {
	a := hm.TypeVariable('a')
	m := newTensorType(2, a)

	return hm.NewFnType(m, m, m)
}

// vecDotOp is a function with this type:
//		vecDotOp :: (Float a) ⇒ Vector a → Vector a → a
//
// For the moment only floats are allowed
func vecDotType() hm.Type {
	a := hm.TypeVariable('a')
	v := newTensorType(1, a)

	return hm.NewFnType(v, v, a)
}

// outerProdOp is a function with this type:
//		outerProdOp :: (Float a) ⇒ Vector a → Vector a → Matrix a
//
// For the moment only floats are allowed
func outerProdType() hm.Type {
	a := hm.TypeVariable('a')
	v := newTensorType(1, a)
	m := newTensorType(2, a)

	return hm.NewFnType(v, v, m)
}
