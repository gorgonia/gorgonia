package gorgonia

// ā and Ā are used to denote that it's a matrix/vector type.
// if you want to type it, it's Latin Letter A with Macron (lowercase and capital)
// Codepoints : U+101 for the small one, and U+100 for the capital one

type āBinaryOperator byte

const (
	matMulOperator    āBinaryOperator = iota // emits S/DGEMM BLAS calls
	matVecMulOperator                        // emits S/DGEMV BLAS calls
	vecDotOperator                           // emits S/DDOT BLAS calls
	outerProdOperator                        // emits S/DGER BLAS calls

	maxĀBinaryOperator // delimits all possible linalg operators. Add above this line
)

func (op āBinaryOperator) String() string {
	if op >= maxĀBinaryOperator {
		return "UNSUPPORTED LINEAR ALGEBRA OPERATOR"
	}
	return āBinOpStrs[op]
}

func (op āBinaryOperator) Type() Type {
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
		if dzdx, err = binOpNode(op, y, z); err == nil {
			dzdy, err = binOpNode(op, z, x)
		}
	case !transA && transB:
		if dzdx, err = binOpNode(op, z, y); err == nil {
			op.transA = true
			dzdy, err = binOpNode(op, z, x)
		}
	case transA && !transB:
		op.transB = true
		if dzdx, err = binOpNode(op, y, z); err == nil {
			op.transB = false
			dzdy, err = binOpNode(op, x, z)
		}
	case !transA && !transB:
		op.transB = true
		if dzdx, err = binOpNode(op, z, y); err == nil {
			op.transA = true
			op.transB = false
			dzdy, err = binOpNode(op, x, z)
		}
	}
	retVal = Nodes{dzdx, dzdy}
	return
}

func matMulDiff(transA, transB bool, x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	op := linAlgBinOp{
		āBinaryOperator: matMulOperator,
	}

	switch {
	case transA && transB:
		op.transA = transA
		op.transB = transB

		// dzdx
		err = op.IncrDo(xdv.d, ydv.Value, zdv.Value)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		} else if err != nil {
			return
		}

		// dzdy
		err = op.IncrDo(ydv.d, zdv.Value, xdv.Value)
		if ver, ok := err.(Valuer); ok {
			ydv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
		}
		return

	case !transA && transB:
		// dzdx
		err = op.IncrDo(xdv.d, zdv.Value, ydv.Value)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		} else if err != nil {
			return
		}

		// dzdy
		op.transA = true
		err = op.IncrDo(ydv.d, zdv.Value, xdv.Value)
		if ver, ok := err.(Valuer); ok {
			ydv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
		}
		return

	case transA && !transB:
		// dzdx
		op.transB = true
		err = op.IncrDo(xdv.d, ydv.Value, zdv.Value)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
		}

		// dzdy
		op.transA = false
		op.transB = false
		err = op.IncrDo(ydv.d, xdv.Value, zdv.Value)
		if ver, ok := err.(Valuer); ok {
			ydv.SetDeriv(ver.Value()) // ignore errors on purpose
		} else if err != nil {
			return
		}

		return
	case !transA && !transB:
		op.transB = true
		err = op.IncrDo(xdv.d, zdv.Value, ydv.Value)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		} else if err != nil {
			return
		}

		op.transA = true
		op.transB = false
		err = op.IncrDo(ydv.d, xdv.Value, zdv.Value)
		if ver, ok := err.(Valuer); ok {
			ydv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
		}
		return
	}

	panic("unreachable")
}

func matVecMulDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = OuterProd(gradZ, y); err == nil {
		op := linAlgBinOp{
			āBinaryOperator: matVecMulOperator,
			transA:          !transA,
		}

		if dzdy, err = binOpNode(op, x, gradZ); err == nil {
			retVal = Nodes{dzdx, dzdy}
		}
	}
	return
}

func matVecMulDiff(transA, transB bool, x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	op := linAlgBinOp{
		āBinaryOperator: outerProdOperator,
	}

	err = op.IncrDo(xdv.d, zdv.d, ydv.Value)
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	} else if err != nil {
		return
	}

	op = linAlgBinOp{
		āBinaryOperator: matVecMulOperator,
		transA:          !transA,
	}

	err = op.IncrDo(ydv.d, xdv.Value, zdv.d)
	if ver, ok := err.(Valuer); ok {
		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func vecDotDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = Mul(y, gradZ); err == nil {
		if dzdy, err = Mul(x, gradZ); err == nil {
			retVal = Nodes{dzdx, dzdy}
		}
	}
	return
}

func vecDotDiff(transA, transB bool, x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	mul := newElemBinOp(mulOpType, x, z)
	err = mul.IncrDo(xdv.d, ydv.Value, zdv.d)
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	} else if err != nil {
		return
	}

	err = mul.IncrDo(ydv.d, xdv.Value, zdv.d)
	if ver, ok := err.(Valuer); ok {
		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func outerProdDiffExpr(transA, transB bool, x, y, z, gradZ *Node) (retVal Nodes, err error) {
	var dzdx, dzdy *Node
	if dzdx, err = Mul(x, gradZ); err == nil {
		if dzdy, err = Mul(y, gradZ); err == nil {
			retVal = Nodes{dzdx, dzdy}
		}
	}
	return
}

func outerProdDiff(transA, transB bool, x, y, z *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)
	zdv := z.boundTo.(*dualValue)

	mul := newElemBinOp(mulOpType, x, z)
	err = mul.IncrDo(xdv.d, xdv.Value, zdv.d)
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
	} else if err != nil {
		return
	}
	err = mul.IncrDo(ydv.d, ydv.Value, zdv.d)
	if ver, ok := err.(Valuer); ok {
		ydv.SetDeriv(ver.Value()) // ignore errors on purpose
		return
	}
	return
}
