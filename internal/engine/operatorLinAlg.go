package engine

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/execution"
	"gorgonia.org/tensor"
)

// ā and Ā are used to denote that it's a matrix/vector type.
// if you want to type it, it's Latin Letter A with Macron (lowercase and capital)
// Codepoints : U+101 for the small one, and U+100 for the capital one

type āBinaryOperator byte

const (
	matMulOperator        āBinaryOperator = iota // emits S/DGEMM BLAS calls
	matVecMulOperator                            // emits S/DGEMV BLAS calls
	vecDotOperator                               // emits S/DDOT BLAS calls
	outerProdOperator                            // emits S/DGER BLAS calls
	batchedMatMulOperator                        // just S/GEMM BLAS calls in a loop

	maxĀBinaryOperator // delimits all possible linalg operators. Add above this line
)

func (op āBinaryOperator) String() string {
	if op >= maxĀBinaryOperator {
		return "UNSUPPORTED LINEAR ALGEBRA OPERATOR"
	}
	return āBinOpStrs[op]
}

func (op āBinaryOperator) Type() hm.Type {
	if op >= maxĀBinaryOperator {
		panic("UNSUPPORTED LINEAR ALGEBRA OPERATOR")
	}
	return āBinOpTypes[op]()
}

func (op āBinaryOperator) DiffWRT(inputs int) []bool {
	if inputs != 2 {
		panic("binary linear algebra operator only supports two and only two inputs")
	}

	if op >= maxĀBinaryOperator {
		panic("Unsupported unary operator is not differentiable")
	}
	return []bool{true, true}
}

// todo: write explanation.
func matMulDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	op := linAlgBinOp{
		āBinaryOperator: matMulOperator,
	}

	switch {
	case transA && transB:
		op.transA = transA
		op.transB = transB
		if dzdx, err = binOpNode(op, y, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
		if dzdy, err = binOpNode(op, gradZ, x); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case !transA && transB:
		if dzdx, err = binOpNode(op, gradZ, y); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}

		op.transA = true
		if dzdy, err = binOpNode(op, gradZ, x); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case transA && !transB:
		op.transB = true
		if dzdx, err = binOpNode(op, y, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}

		op.transB = false
		if dzdy, err = binOpNode(op, x, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case !transA && !transB:
		// dzdy
		op.transA = false
		op.transB = true
		if dzdx, err = binOpNode(op, gradZ, y); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
		// do dzdx
		op.transA = true
		op.transB = false
		if dzdy, err = binOpNode(op, x, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	}
	retVal = Nodes{dzdx, dzdy}
	return
}

func matMulDiff(ctx execution.Context, transA, transB bool, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	op := linAlgBinOp{
		āBinaryOperator: matMulOperator,
	}

	switch {
	case transA && transB:
		op.transA = transA
		op.transB = transB

		// dzdx
		err = op.IncrDo(xdv.D, ydv.Value, zdv.D)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		err = op.IncrDo(ydv.D, zdv.D, xdv.Value)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, y)
		}

		return

	case !transA && transB:
		// dzdx
		err = op.IncrDo(xdv.D, zdv.D, ydv.Value)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		op.transA = true
		err = op.IncrDo(ydv.D, zdv.D, xdv.Value)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		return

	case transA && !transB:
		// dzdx
		op.transB = true
		err = op.IncrDo(xdv.D, ydv.Value, zdv.D)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		op.transA = false
		op.transB = false
		err = op.IncrDo(ydv.D, xdv.Value, zdv.D)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
		return
	case !transA && !transB:
		op.transB = true
		err = op.IncrDo(xdv.D, zdv.D, ydv.Value)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		op.transA = true
		op.transB = false
		err = op.IncrDo(ydv.D, xdv.Value, zdv.D)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
		return
	}

	panic("unreachable")
}

func matVecMulDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if transA {
		dzdx, err = OuterProd(y, gradZ)
	} else {
		dzdx, err = OuterProd(gradZ, y)
	}

	if err != nil {
		return nil, errors.Wrap(err, "Failed to carry outper product")
	}

	op := linAlgBinOp{
		āBinaryOperator: matVecMulOperator,
		transA:          !transA,
	}

	if dzdy, err = binOpNode(op, x, gradZ); err != nil {
		return nil, errors.Wrapf(err, binOpNodeFail, op)
	}
	return Nodes{dzdx, dzdy}, nil
}

func matVecMulDiff(ctx execution.Context, transA, transB bool, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	op := linAlgBinOp{
		āBinaryOperator: outerProdOperator,
	}

	if transA {
		err = op.IncrDo(xdv.D, ydv.Value, zdv.D)
	} else {
		err = op.IncrDo(xdv.D, zdv.D, ydv.Value)
	}
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	op = linAlgBinOp{
		āBinaryOperator: matVecMulOperator,
		transA:          !transA,
	}

	err = op.IncrDo(ydv.D, xdv.Value, zdv.D)
	if err = checkErrSetDeriv(err, ydv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func vecDotDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = HadamardProd(y, gradZ); err == nil {
		if dzdy, err = HadamardProd(x, gradZ); err == nil {
			retVal = Nodes{dzdx, dzdy}
		} else {
			return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
		}
	} else {
		return nil, errors.Wrap(err, "Failed to carry HadamardProd()")
	}
	return
}

func vecDotDiff(ctx execution.Context, transA, transB bool, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	mul := newElemBinOp(mulOpType, x, z)
	err = mul.IncrDo(xdv.D, ydv.Value, zdv.D)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	err = mul.IncrDo(ydv.D, xdv.Value, zdv.D)
	if err = checkErrSetDeriv(err, ydv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func outerProdDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = Mul(x, gradZ); err == nil {
		if dzdy, err = Mul(y, gradZ); err == nil {
			retVal = Nodes{dzdx, dzdy}
		} else {
			return nil, errors.Wrap(err, "Failed to carry Mul()")
		}
	} else {
		return nil, errors.Wrap(err, "Failed to carry Mul()")
	}
	return
}

func outerProdDiff(ctx execution.Context, transA, transB bool, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	mul := newElemBinOp(mulOpType, x, z)
	err = mul.IncrDo(xdv.D, xdv.Value, zdv.D)
	err = mul.IncrDo(xdv.D, ydv.Value, zdv.D)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	err = mul.IncrDo(ydv.D, ydv.Value, zdv.D)
	if err = checkErrSetDeriv(err, ydv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func batchedMatMulDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	op := linAlgBinOp{
		āBinaryOperator: batchedMatMulOperator,
	}

	switch {
	case transA && transB:
		op.transA = transA
		op.transB = transB
		if dzdx, err = binOpNode(op, y, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
		if dzdy, err = binOpNode(op, gradZ, x); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case !transA && transB:
		if dzdx, err = binOpNode(op, gradZ, y); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}

		op.transA = true
		if dzdy, err = binOpNode(op, gradZ, x); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case transA && !transB:
		op.transB = true
		if dzdx, err = binOpNode(op, y, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}

		op.transB = false
		if dzdy, err = binOpNode(op, x, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	case !transA && !transB:
		// dzdy
		op.transA = false
		op.transB = true
		if dzdx, err = binOpNode(op, gradZ, y); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
		// do dzdx
		op.transA = true
		op.transB = false
		if dzdy, err = binOpNode(op, x, gradZ); err != nil {
			return nil, errors.Wrapf(err, binOpNodeFail, op)
		}
	}
	retVal = Nodes{dzdx, dzdy}
	return
}

func batchedMatMulDiff(ctx execution.Context, transA, transB bool, x, y, z *Node) (err error) {
	xdv, ydv, zdv := getDV3(x, y, z)

	op := linAlgBinOp{
		āBinaryOperator: batchedMatMulOperator,
	}

	switch {
	case transA && transB:
		op.transA = transA
		op.transB = transB

		// dzdx
		err = op.IncrDo(xdv.D, ydv.Value, zdv.D)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		err = op.IncrDo(ydv.D, zdv.D, xdv.Value)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, y)
		}

		return

	case !transA && transB:
		// dzdx
		err = op.IncrDo(xdv.D, zdv.D, ydv.Value)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		op.transA = true
		err = op.IncrDo(ydv.D, zdv.D, xdv.Value)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		return

	case transA && !transB:
		// dzdx
		op.transB = true
		err = op.IncrDo(xdv.D, ydv.Value, zdv.D)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		// dzdy
		op.transA = false
		op.transB = false
		err = op.IncrDo(ydv.D, xdv.Value, zdv.D)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
		return
	case !transA && !transB:
		op.transB = true
		err = op.IncrDo(xdv.D, zdv.D, ydv.Value)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}

		op.transA = true
		op.transB = false
		err = op.IncrDo(ydv.D, xdv.Value, zdv.D)
		if err = checkErrSetDeriv(err, ydv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
		return
	}

	panic("unreachable")
}

func batchedMatMul(a, b, c tensor.Tensor, transA, transB, incr bool) (retVal tensor.Tensor, err error) {
	shapeA := a.Shape()
	shapeB := b.Shape()

	batchSize := shapeA[0]

	if c == nil {
		c = tensor.New(tensor.Of(a.Dtype()), tensor.WithShape(batchSize, shapeA[2], shapeB[1]), tensor.WithEngine(a.Engine()))
	}

	var as, bs, cs tensor.Tensor
	for i := 0; i < batchSize; i++ {
		if as, err = a.Slice(S(i)); err != nil {

		}
		if bs, err = b.Slice(S(i)); err != nil {

		}
		if cs, err = c.Slice(S(i)); err != nil {

		}

		if transA {
			as.T()
		}
		if transB {
			bs.T()
		}

		var fo tensor.FuncOpt
		if incr {
			fo = tensor.WithIncr(cs)
		} else {
			fo = tensor.WithReuse(cs)
		}

		if _, err = tensor.MatMul(as, bs, fo); err != nil {

		}
	}

	return c, nil
}
