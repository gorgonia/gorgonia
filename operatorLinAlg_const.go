package gorgonia

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

var āBinOpTypes = [maxĀBinaryOperator]func() Type{
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
func matVecMulType() Type {
	a := newTypeVariable("a", withTVConstraints(floats))
	v := newTensorType(1, a)
	m := newTensorType(2, a)

	return newFunctionType(m, v, v)
}

// matMulOp is a function with this type:
//		matMulOp :: (Float a) ⇒ Matrix a → Matrix a → Matrix a
//
// For the moment only floats are allowed
func matMulType() Type {
	a := newTypeVariable("a", withTVConstraints(floats))
	m := newTensorType(2, a)

	return newFunctionType(m, m, m)
}

// vecDotOp is a function with this type:
//		vecDotOp :: (Float a) ⇒ Vector a → Vector a → a
//
// For the moment only floats are allowed
func vecDotType() Type {
	a := newTypeVariable("a", withTVConstraints(floats))
	v := newTensorType(1, a)

	return newFunctionType(v, v, a)
}

// outerProdOp is a function with this type:
//		outerProdOp :: (Float a) ⇒ Vector a → Vector a → Matrix a
//
// For the moment only floats are allowed
func outerProdType() Type {
	a := newTypeVariable("a", withTVConstraints(floats))
	v := newTensorType(1, a)
	m := newTensorType(2, a)

	return newFunctionType(v, v, m)
}
