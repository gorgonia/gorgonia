package gorgonia

import (
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
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

func (f *sf32UnaryOperator) unaryOpType() ʘUnaryOperatorType {
	switch f {
	case &absf32:
		return absOpType
	case &signf32:
		return signOpType
	case &ceilf32:
		return ceilOpType
	case &floorf32:
		return floorOpType
	case &sinf32:
		return sinOpType
	case &cosf32:
		return cosOpType
	case &expf32:
		return expOpType
	case &lnf32:
		return lnOpType
	case &log2f32:
		return log2OpType
	case &negf32:
		return negOpType
	case &squaref32:
		return squareOpType
	case &sqrtf32:
		return sqrtOpType
	case &inversef32:
		return inverseOpType
	case &cubef32:
		return cubeOpType
	case &tanhf32:
		return tanhOpType
	case &sigmoidf32:
		return sigmoidOpType
	case &log1pf32:
		return log1pOpType
	case &expm1f32:
		return expm1OpType
	case &softplusf32:
		return softplusOpType
	}
	return maxʘUnaryOperator
}

func (f *sf32UnaryOperator) String() string { return f.unaryOpType().String() }

type sf64UnaryOperator func(float64) float64

func (f *sf64UnaryOperator) unaryOpType() ʘUnaryOperatorType {
	switch f {
	case &absf64:
		return absOpType
	case &signf64:
		return signOpType
	case &ceilf64:
		return ceilOpType
	case &floorf64:
		return floorOpType
	case &sinf64:
		return sinOpType
	case &cosf64:
		return cosOpType
	case &expf64:
		return expOpType
	case &lnf64:
		return lnOpType
	case &log2f64:
		return log2OpType
	case &negf64:
		return negOpType
	case &squaref64:
		return squareOpType
	case &sqrtf64:
		return sqrtOpType
	case &inversef64:
		return inverseOpType
	case &cubef64:
		return cubeOpType
	case &tanhf64:
		return tanhOpType
	case &sigmoidf64:
		return sigmoidOpType
	case &log1pf64:
		return log1pOpType
	case &expm1f64:
		return expm1OpType
	case &softplusf64:
		return softplusOpType
	}

	return maxʘUnaryOperator
}

func (f *sf64UnaryOperator) String() string { return f.unaryOpType().String() }

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
	if retVal, err = Sign(x); err == nil {
		WithGroupName(gradClust)(retVal)
		retVal, err = HadamardProd(gradY, retVal)
		if err != nil {
			return nil, errors.Wrap(err, hadamardProdFail)
		}
	} else {
		return nil, errors.Wrap(err, "Failed to call Sign()")
	}
	return
}

func absDiff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	sign := newElemUnaryOp(signOpType, x)

	var d Value
	if d, err = sign.Do(xdv.Value); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		mul := newElemBinOp(mulOpType, y, x)
		err = mul.IncrDo(xdv.d, d, ydv.d)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	cos := newElemUnaryOp(cosOpType, x)

	var d Value
	if d, err = cos.Do(xdv.Value); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		mul := newElemBinOp(mulOpType, x, y)
		err = mul.IncrDo(xdv.d, d, ydv.d)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

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
			if ver, ok := err.(Valuer); ok {
				xdv.SetDeriv(ver.Value()) // ignore errors on purpose
				return nil
			}

		}
	}
	return
}

func expDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return HadamardProd(y, gradY)
}

func expDiff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	mul := newElemBinOp(mulOpType, x, y)
	err = mul.IncrDo(xdv.d, ydv.Value, ydv.d)
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

// solution is 1/x.
// Upon multiplying with gradY for chain rule, it simply becomes gradY/x
func lnDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return HadamardDiv(gradY, x)
}

func lnDiff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	div := newElemBinOp(divOpType, y, x)

	err = div.IncrDo(xdv.d, ydv.d, xdv.Value)
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}

	return
}

// 1/(x*ln(2))
func log2DiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var log2 *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		log2 = ln2f32
	case Float64:
		log2 = ln2f64
	default:
		return nil, errors.Errorf("log2DiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
	}

	if retVal, err = HadamardProd(x, log2); err == nil {
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

func log2Diff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var log2 *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		log2 = ln2f32
	case Float64:
		log2 = ln2f64
	default:
		return errors.Errorf("log2DiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}

	return
}

func negDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	return Neg(gradY)
}

func negDiff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	sub := newElemBinOp(subOpType, x, y)
	_, err = sub.UnsafeDo(xdv.d, ydv.d)
	if ver, ok := err.(Valuer); ok {
		return xdv.SetDeriv(ver.Value())
	}
	return
}

func squareDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var two *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		two = twof32
	case Float64:
		two = twof64
	default:
		return nil, errors.Errorf("squareDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
	}

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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var two *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		two = twof32
	case Float64:
		two = twof64
	default:
		return errors.Errorf("squareDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
	}

	mul := newElemBinOp(mulOpType, x, y)

	var d Value
	if d, err = mul.Do(xdv.Value, two.boundTo); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		err = mul.IncrDo(xdv.d, d, ydv.d)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
		}
	}
	return
}

func sqrtDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var two *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		two = twof32
	case Float64:
		two = twof64
	default:
		return nil, errors.Errorf("sqrtDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var two *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		two = twof32
	case Float64:
		two = twof64
	default:
		return errors.Errorf("sqrtDiff does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
	}

	mul := newElemBinOp(mulOpType, x, y)

	var d Value
	if d, err = mul.Do(ydv.Value, two.boundTo); err == nil {
		if dT, ok := d.(tensor.Tensor); ok {
			defer returnTensor(dT)
		}

		div := newElemBinOp(divOpType, y, x)
		err = div.IncrDo(xdv.d, ydv.d, d)
		if ver, ok := err.(Valuer); ok {
			xdv.SetDeriv(ver.Value()) // ignore errors on purpose
			return nil
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func cubeDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var three *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		three = threef32
	case Float64:
		three = threef64
	default:
		return nil, errors.Errorf("cubeDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var three *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		three = threef32
	case Float64:
		three = threef64
	default:
		return errors.Errorf("cubeDiff does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func tanhDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return nil, errors.Errorf("tanhDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return errors.Errorf("tanhDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func sigmoidDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return nil, errors.Errorf("tanhDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return errors.Errorf("tanhDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

// 1/(x+1)
func log1pDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return nil, errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return nil, errors.Errorf("log1pDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

	var one *Node
	var dt Dtype

	if dt, err = dtypeOf(x.t); err != nil {
		return errors.Wrap(err, dtypeOfFail)
	}

	switch dt {
	case Float32:
		one = onef32
	case Float64:
		one = onef64
	default:
		return errors.Errorf("log1pDiffExpr does not handle Dtypes other than Float32 and Float64. Got %v instead", dt)
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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func expm1DiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Exp(x); err == nil {
		WithGroupName(gradClust)(retVal)
		return HadamardProd(gradY, retVal)
	} else {
		return nil, errors.Wrap(err, "Failled to carry Exp()")
	}
}

func expm1Diff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}

func softplusDiffExpr(x, y, gradY *Node) (retVal *Node, err error) {
	if retVal, err = Sigmoid(x); err == nil {
		WithGroupName(gradClust)(retVal)
		return HadamardProd(retVal, gradY)
	} else {
		return nil, errors.Wrap(err, "Failed to carry Sigmoid()")
	}
}

func softplusDiff(x, y *Node) (err error) {
	xdv := x.boundTo.(*dualValue)
	ydv := y.boundTo.(*dualValue)

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
	if ver, ok := err.(Valuer); ok {
		xdv.SetDeriv(ver.Value()) // ignore errors on purpose
		return nil
	}
	return
}
