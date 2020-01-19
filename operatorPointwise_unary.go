package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// a ʘUnaryOperator is essentially a function that takes a float32 or float64 and returns the same
// pros : no overloading = clear understanding
// cons : no overloading = a lot of extra code
//
// There are TWO ʘUnaryOperator types so far:
//		sf32UnaryOperator - scalar float32 unary operator
//		sf64UnaryOperator - scalar float64 unary operator
//
// Because TensorTypes are parameterized by a scalar type, it isn't necessary to create operators
// that will work on TensorTypes. A simple type switch will do.
//
// n.b.: ʘ is used to denote pointwiseness of the operator.
// if you want to type it, it's U+0298 - Latin Letter Bilabial Click
type ʘUnaryOperator interface {
	unaryOpType() ʘUnaryOperatorType
	String() string
}

type sf32UnaryOperator func(float32) float32
type sf64UnaryOperator func(float64) float64

// unaryCheckApply checks in a interface is fulfilled. If it is, that engine is used instead
func unaryCheckApply(op ʘUnaryOperator, t tensor.Tensor, opts ...tensor.FuncOpt) (retVal tensor.Tensor, err error) {
	e := t.Engine()
	switch op.unaryOpType() {
	case absOpType:
		if oe, ok := e.(tensor.Abser); ok {
			return oe.Abs(t, opts...)
		}
	case signOpType:
		if oe, ok := e.(tensor.Signer); ok {
			return oe.Sign(t, opts...)
		}
	case ceilOpType:
	case floorOpType:
	case sinOpType:
	case cosOpType:
	case expOpType:
		if oe, ok := e.(tensor.Exper); ok {
			return oe.Exp(t, opts...)
		}
	case lnOpType:
		if oe, ok := e.(tensor.Loger); ok {
			return oe.Log(t, opts...)
		}
	case log2OpType:
		if oe, ok := e.(tensor.Log2er); ok {
			return oe.Log2(t, opts...)
		}
	case negOpType:
		if oe, ok := e.(tensor.Neger); ok {
			return oe.Neg(t, opts...)
		}
	case squareOpType:
		if oe, ok := e.(tensor.Squarer); ok {
			return oe.Square(t, opts...)
		}
	case sqrtOpType:
		if oe, ok := e.(tensor.Sqrter); ok {
			return oe.Sqrt(t, opts...)
		}
	case inverseOpType:
		if oe, ok := e.(tensor.Inver); ok {
			return oe.Inv(t, opts...)
		}
	case inverseSqrtOpType:
		if oe, ok := e.(tensor.InvSqrter); ok {
			return oe.InvSqrt(t, opts...)
		}
	case cubeOpType:
		if oe, ok := e.(tensor.Cuber); ok {
			return oe.Cube(t, opts...)
		}
	case tanhOpType:
		if oe, ok := e.(tensor.Tanher); ok {
			return oe.Tanh(t, opts...)
		}
	case sigmoidOpType:
	case log1pOpType:
	case expm1OpType:
	case softplusOpType:
	}

	//default case:
	var fn interface{}
	switch opFn := op.(type) {
	case *sf64UnaryOperator:
		fn = (func(float64) float64)(*opFn)
	case *sf32UnaryOperator:
		fn = (func(float32) float32)(*opFn)
	}

	return t.Apply(fn, opts...)
}

/*
DIFFERENTIATION EXPRESSIONS

All the functions here are expressed in terms of *Node and/or Nodes

*/

func nondiffUnaryOpExpr(x, y, gradY *Node) (*Node, error) {
	return nil, errors.Errorf("Nondifferentiable Function")
}
func nondiffUnaryOp(x, y *Node) error {
	return AutoDiffError{}
}

// apparently abs is differentiable
func absDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Sign(x); err != nil {
		return nil, errors.Wrap(err, "Failed to call Sign()")
	}
	WithGroupName(gradClust)(retVal)

	if retVal, err = HadamardProd(gradY, retVal); err != nil {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func absDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var d Value
	sign := newElemUnaryOp(signOpType, x)
	if d, err = sign.Do(xdv.Value); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		mul := newElemBinOp(mulOpType, y, x)
		err = mul.IncrDo(xdv.d, d, ydv.d)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
	}
	return
}

// Solution here
// https://www.symbolab.com/solver/step-by-step/%5Cfrac%7Bd%7D%7Bdx%7D%5Cleft(sin%5Cleft(x%5Cright)%5Cright)
func sinDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Cos(x); err == nil {
		WithGroupName(gradClust)(retVal)
		retVal, err = HadamardProd(retVal, gradY)
		if err != nil {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
	} else {
		return nil, errors.Wrap(err, "Failed to carry Cos()")
	}
	return
}

func sinDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	cos := newElemUnaryOp(cosOpType, x)

	var d Value
	if d, err = cos.Do(xdv.Value); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		mul := newElemBinOp(mulOpType, x, y)
		err = mul.IncrDo(xdv.d, d, ydv.d)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
	}
	return
}

// Solution here (then apply chain rule to result by multiplying gradY):
// https://www.symbolab.com/solver/step-by-step/%5Cfrac%7Bd%7D%7Bdx%7D%5Cleft(cos%5Cleft(x%5Cright)%5Cright)
func cosDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Sin(x); err == nil {
		WithGroupName(gradClust)(retVal)
		if retVal, err = Neg(retVal); err == nil {
			WithGroupName(gradClust)(retVal)
			retVal, err = HadamardProd(retVal, gradY)
			if err != nil {
				return nil, errors.Wrap(err, hadamardProdFail)
			}
		} else {
			return nil, errors.Wrap(err, negFail)
		}
	} else {
		return nil, errors.Wrap(err, "Failed to call Sin()")
	}
	return
}

func cosDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	sin := newElemUnaryOp(sinOpType, x)

	var d Value
	if d, err = sin.Do(xdv.Value); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		neg := newElemUnaryOp(negOpType, x)
		if d, err = neg.UnsafeDo(d); err == nil {
			mul := newElemBinOp(mulOpType, x, y)
			err = mul.IncrDo(xdv.d, d, ydv.d)
			if err = checkErrSetDeriv(err, xdv); err != nil {
				return errors.Wrapf(err, autodiffFail, x)
			}

		}
	}
	return
}

func expDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return HadamardProd(y, gradY)
}

func expDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	mul := newElemBinOp(mulOpType, x, y)
	err = mul.IncrDo(xdv.d, ydv.Value, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

// solution is 1/x.
// Upon multiplying with gradY for chain rule, it simply becomes gradY/x
func lnDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return HadamardDiv(gradY, x)
}

func lnDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	div := newElemBinOp(divOpType, y, x)

	err = div.IncrDo(xdv.d, ydv.d, xdv.Value)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	return
}

// 1/(x*ln(2))
func log2DiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var log2 *Node
	if log2, err = getConst(x, "log2"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = HadamardDiv(x, log2); err != nil {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	WithGroupName(gradClust)(retVal)
	if retVal, err = HadamardDiv(gradY, retVal); err != nil {
		return nil, errors.Wrap(err, hadamardDivFail)
	}
	return
}

func log2Diff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var log2 *Node
	if log2, err = getConst(x, "log2"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	mul := newElemBinOp(mulOpType, x, log2)
	var d Value
	if d, err = mul.Do(xdv.Value, log2.boundTo); err != nil {
		return errors.Wrapf(err, doFail, mul)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	div := newElemBinOp(divOpType, y, x)
	err = div.IncrDo(xdv.d, ydv.d, d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	return
}

func negDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return Neg(gradY)
}

func negDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	sub := newElemBinOp(subOpType, x, y)
	var d Value
	d, err = sub.UnsafeDo(xdv.d, ydv.d)

	// first we check if what essentially is a noIncrError is called
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}

	// then we set derivs, if d is a scalar
	if _, ok := xdv.Value.(Scalar); ok {
		if err = xdv.SetDeriv(d); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
	}

	return
}

func squareDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	// symdiffLogf("X %v and TWO %v", x.Shape(), two.Shape())
	if retVal, err = HadamardProd(x, two); err == nil {
		symdiffLogf("Spawned: %d", retVal.ID())
		WithGroupName(gradClust)(retVal)
		retVal, err = HadamardProd(retVal, gradY)
		if err != nil {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
		symdiffLogf("Spawned: %d", retVal.ID())
	} else {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func squareDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	var d Value
	mul := newElemBinOp(mulOpType, x, y)
	if d, err = mul.Do(xdv.Value, two.boundTo); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		err = mul.IncrDo(xdv.d, d, ydv.d)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
	}
	return
}

func sqrtDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = HadamardProd(two, y); err == nil {
		WithGroupName(gradClust)(retVal)
		retVal, err = HadamardDiv(gradY, retVal)
		if err != nil {
			return nil, errors.Wrap(err, hadamardDivFail)
		}
	} else {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func sqrtDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	mul := newElemBinOp(mulOpType, x, y)

	var d Value
	if d, err = mul.Do(ydv.Value, two.boundTo); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		div := newElemBinOp(divOpType, y, x)
		err = div.IncrDo(xdv.d, ydv.d, d)
		if err = checkErrSetDeriv(err, xdv); err != nil {
			return errors.Wrapf(err, autodiffFail, x)
		}
	}
	return
}

func inverseDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = HadamardProd(y, y); err == nil {
		WithGroupName(gradClust)(retVal)
		if retVal, err = Neg(retVal); err == nil {
			WithGroupName(gradClust)(retVal)
			retVal, err = HadamardProd(retVal, gradY)
			if err != nil {
				return nil, errors.Wrap(err, hadamardProdFail)
			}
		} else {
			return nil, errors.Wrap(err, negFail)
		}
	} else {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func inverseDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	sq := newElemUnaryOp(squareOpType, y)

	var d Value
	if d, err = sq.Do(ydv.Value); err != nil {
		return errors.Wrapf(err, doFail, sq)
	}

	neg := newElemUnaryOp(negOpType, y)
	if d, err = neg.Do(d); err != nil {
		return errors.Wrapf(err, doFail, neg)
	}
	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	mul := newElemBinOp(mulOpType, y, y)
	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func inverseSqrtDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}
	if retVal, err = Cube(y); err != nil {
		return nil, errors.Wrapf(err, cubeFail)
	}
	if retVal, err = HadamardProd(two, retVal); err != nil {
		return nil, errors.Wrapf(err, hadamardProdFail)
	}
	if retVal, err = HadamardDiv(gradY, retVal); err != nil {
		return nil, errors.Wrapf(err, hadamardDivFail)
	}
	return Neg(retVal)
}

func inverseSqrtDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)
	var two *Node
	if two, err = getConst(x, "two"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	cb := newElemUnaryOp(cubeOpType, y)
	var d Value
	if d, err = cb.Do(ydv.Value); err != nil {
		return errors.Wrapf(err, doFail, cb)
	}

	mul := newElemBinOp(mulOpType, x, y)
	if d, err = mul.Do(two.boundTo, d); err != nil {
		return errors.Wrapf(err, doFail, mul)
	}

	div := newElemBinOp(divOpType, y, x)
	if d, err = div.Do(ydv.d, d); err != nil {
		return errors.Wrapf(err, doFail, div)
	}

	sub := newElemBinOp(subOpType, x, y)
	if _, err = sub.Do(xdv.d, d); err != nil {
		return errors.Wrapf(err, doFail, sub)
	}
	return nil
}

func cubeDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var three *Node
	if three, err = getConst(x, "three"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = HadamardProd(x, x); err == nil {
		WithGroupName(gradClust)(retVal)
		if retVal, err = HadamardProd(retVal, three); err == nil {
			WithGroupName(gradClust)(retVal)
			retVal, err = HadamardProd(retVal, gradY)
			if err != nil {
				return nil, errors.Wrap(err, hadamardProdFail)
			}
		} else {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
	} else {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func cubeDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var three *Node
	if three, err = getConst(x, "three"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	mul := newElemBinOp(mulOpType, x, y)
	var d Value
	if d, err = mul.Do(xdv.Value, xdv.Value); err != nil {
		return errors.Wrapf(err, doFail, mul)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	if d, err = mul.UnsafeDo(d, three.boundTo); err != nil {
		return errors.Wrapf(err, unsafeDoFail, mul)
	}

	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func tanhDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = HadamardProd(y, y); err == nil {
		WithGroupName(gradClust)(retVal)
		if retVal, err = Sub(one, retVal); err == nil {
			WithGroupName(gradClust)(retVal)
			retVal, err = HadamardProd(retVal, gradY)
			if err != nil {
				return nil, errors.Wrap(err, hadamardProdFail)
			}
		} else {
			return nil, errors.Wrap(err, subFail)
		}
	} else {
		return nil, errors.Wrap(err, hadamardProdFail)
	}
	return
}

func tanhDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	sq := newElemUnaryOp(squareOpType, y)

	var d Value
	if d, err = sq.Do(ydv.Value); err != nil {
		return errors.Wrapf(err, doFail, sq)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	sub := newElemBinOp(subOpType, one, y)
	if d, err = sub.UnsafeDo(one.boundTo, d); err != nil {
		return errors.Wrapf(err, unsafeDoFail, sub)
	}

	mul := newElemBinOp(mulOpType, x, y)
	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func sigmoidDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = Sub(one, y); err == nil {
		WithGroupName(gradClust)(retVal)
		if retVal, err = HadamardProd(y, retVal); err == nil {
			WithGroupName(gradClust)(retVal)
			retVal, err = HadamardProd(retVal, gradY)
			if err != nil {
				return nil, errors.Wrap(err, hadamardProdFail)
			}
		} else {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
	} else {
		return nil, errors.Wrap(err, subFail)
	}
	return
}

func sigmoidDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	sub := newElemBinOp(subOpType, one, y)

	var d Value
	if d, err = sub.Do(one.boundTo, ydv.Value); err != nil {
		return errors.Wrapf(err, doFail, sub)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	mul := newElemBinOp(mulOpType, x, y)
	if d, err = mul.UnsafeDo(d, ydv.Value); err != nil {
		return errors.Wrapf(err, unsafeDoFail, mul)
	}

	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

// 1/(x+1)
func log1pDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return nil, errors.Wrap(err, "getConst failed")
	}

	if retVal, err = Add(x, one); err == nil {
		WithGroupName(gradClust)(retVal)
		retVal, err = HadamardDiv(gradY, retVal)
		if err != nil {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
	} else {
		return nil, errors.Wrap(err, "Failed to carry Add()")
	}
	return
}

func log1pDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	var one *Node
	if one, err = getConst(x, "one"); err != nil {
		return errors.Wrap(err, "getConst failed")
	}

	add := newElemBinOp(addOpType, x, one)

	var d Value
	if d, err = add.Do(xdv.Value, one.boundTo); err != nil {
		return errors.Wrapf(err, doFail, add)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	div := newElemBinOp(divOpType, y, x)
	err = div.IncrDo(xdv.d, ydv.d, d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func expm1DiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Exp(x); err == nil {
		WithGroupName(gradClust)(retVal)
		return HadamardProd(gradY, retVal)
	}
	return nil, errors.Wrap(err, "Failled to carry Exp()")
}

func expm1Diff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	exp := newElemUnaryOp(expOpType, x)

	var d Value
	if d, err = exp.Do(xdv.Value); err != nil {
		return errors.Wrapf(err, doFail, exp)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	mul := newElemBinOp(mulOpType, x, y)
	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}

func softplusDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Sigmoid(x); err == nil {
		WithGroupName(gradClust)(retVal)
		return HadamardProd(retVal, gradY)
	}
	return nil, errors.Wrap(err, "Failed to carry Sigmoid()")
}

func softplusDiff(x, y *Node) (err error) {
	xdv, ydv := getDV(x, y)

	sigmoid := newElemUnaryOp(sigmoidOpType, x)

	var d Value
	if d, err = sigmoid.Do(xdv.Value); err != nil {
		return errors.Wrapf(err, doFail, sigmoid)
	}

	if dT, ok := d.(tensor.Tensor); ok {
		defer returnTensor(dT)
	}

	mul := newElemBinOp(mulOpType, x, y)
	err = mul.IncrDo(xdv.d, d, ydv.d)
	if err = checkErrSetDeriv(err, xdv); err != nil {
		return errors.Wrapf(err, autodiffFail, x)
	}
	return
}
