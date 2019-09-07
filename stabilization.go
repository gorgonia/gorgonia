package gorgonia

import "github.com/pkg/errors"

var unaryOpStabilizationFns = make(map[ʘUnaryOperatorType][]func(*Node) (*Node, error))
var binOpStabilizationFns = make(map[ʘBinaryOperatorType][]func(*Node, *Node) (*Node, error))

func init() {
	unaryOpStabilizationFns[lnOpType] = []func(*Node) (*Node, error){logSigmoidStabilization, logStabilization}
	binOpStabilizationFns[subOpType] = []func(*Node, *Node) (*Node, error){expStabilization}
	unaryOpStabilizationFns[log1pOpType] = []func(*Node) (*Node, error){log1pExpStabilization, log1pNegSigmoidStabilization}
}

// logStabilization converts log(1+a) and log(a+1) to log1p(a) and log(1-a) to log1p(-a)
// place before log; a should be +
func logStabilization(a *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing log(1+a) of %v", a)
	enterLogScope()
	defer leaveLogScope()

	var x *Node
	var aop elemBinOp
	var ok bool

	if aop, ok = a.op.(elemBinOp); !ok {
		return a, noStabilizationErr{}
	}
	input0 := a.children[0]
	input1 := a.children[1]

	stabLogf("input0: %v", input0.Name())
	stabLogf("input1: %v", input1.Name())
	bot := aop.ʘBinaryOperator.binOpType()
	switch bot {
	case addOpType:
		if cnst, ok := input0.op.(constant); ok {
			if constEq(cnst, onef32ConstOp) || constEq(cnst, onef64ConstOp) {
				x = input1
				break
			}
		}

		if cnst, ok := input1.op.(constant); ok {
			if constEq(cnst, onef32ConstOp) || constEq(cnst, onef64ConstOp) {
				x = input0
				break
			}
		}

		return a, noStabilizationErr{}
	case subOpType:
		if cnst, ok := input0.op.(constant); !ok || (ok && !constEq(cnst, onef32ConstOp) && !constEq(cnst, onef64ConstOp)) {
			return a, noStabilizationErr{}
		}
		x = input1
	default:
		return a, noStabilizationErr{}
	}

	g := a.g
	g.removeAllEdgesFrom(a) // remove all references
	g.RemoveNode(a)
	defer returnNode(a) // send it back to the pool, since it is literally useless now

	if bot == subOpType {
		if retVal, err = Neg(x); err == nil {
			return Log1p(retVal)
		}
		return nil, errors.Wrap(err, negFail)
	}
	return Log1p(x)
}

// expStabilization converts exp(x)-1 to expm1(x)
// place before sub; i0 should be exp(x); i1 should be 1
func expStabilization(a, b *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing exp(x)-1 to expm1(x) of %v and %v", a, b)
	enterLogScope()
	defer leaveLogScope()

	if cnst, ok := b.op.(constant); !ok || (ok && !constEq(cnst, onef32ConstOp) && !constEq(cnst, onef64ConstOp)) {
		return nil, noStabilizationErr{}
	}

	if euo, ok := a.op.(elemUnaryOp); !ok || euo.unaryOpType() != expOpType {
		return nil, noStabilizationErr{}
	}

	op := newElemUnaryOp(expm1OpType, a.children[0])
	return ApplyOp(op, a.children[0])
}

// oneMinusSigmoidStabilization stabilizes 1-sigmoid(x) by replacing it with sigmoid(-x)
// place before sub
func oneMinusSigmoidStabilization(a, b *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing 1-sigmoid(x) to sigmoid(-x) of %v and %v", a, b)
	enterLogScope()
	defer leaveLogScope()

	if cnst, ok := a.op.(constant); !ok || (ok && !constEq(cnst, onef32ConstOp) && !constEq(cnst, onef64ConstOp)) {
		return nil, noStabilizationErr{}
	}

	if euo, ok := b.op.(elemUnaryOp); !ok || euo.unaryOpType() != sigmoidOpType {
		return nil, noStabilizationErr{}
	}

	x := b.children[0]
	if retVal, err = Neg(x); err == nil {
		return Sigmoid(retVal)
	}
	return nil, errors.Wrap(err, negFail)
}

// logSigmoidStabilization stabilizes log(sigmoid(x)) by replacing it with -softplus(-x)
// place before log; a should be sigmoid(x)
func logSigmoidStabilization(a *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing log sigmoid of %v", a)
	enterLogScope()
	defer leaveLogScope()

	if euo, ok := a.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != sigmoidOpType) {
		return a, noStabilizationErr{}
	}

	x := a.children[0]
	stabLogf("x : %v", x.Name())

	if retVal, err = Neg(x); err == nil {
		if retVal, err = Softplus(retVal); err == nil {
			retVal, err = Neg(retVal)
			if err != nil {
				return nil, errors.Wrap(err, negFail)
			}
			return retVal, nil
		}
		return nil, errors.Wrap(err, softplusFail)
	}
	return nil, errors.Wrap(err, negFail)
}

// log1pExpStabilization stabilizes log1p(exp(x)) by substituting it with softplus(x)
// place before log1p; a should be exp(x)
func log1pExpStabilization(a *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing log1p(exp(x)) of %v", a)
	enterLogScope()
	defer leaveLogScope()

	if euo, ok := a.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != expOpType) {
		stabLogf("op: %v; %v", a.op, a.children)
		return a, noStabilizationErr{}
	}

	x := a.children[0]
	stabLogf("OKKKKK")
	return Softplus(x)
}

// log1pNegSigmoidStabilization stabilizes log1p(-sigmoid(x)) by substituting it with -softplus(x)
// place before log1p;  a should be -sigmoid(x)
func log1pNegSigmoidStabilization(a *Node) (retVal *Node, err error) {
	stabLogf("Stabilizing log1p(-sigmoid(x)) : %v", a)
	enterLogScope()
	defer leaveLogScope()

	if euo, ok := a.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != negOpType) {
		return a, noStabilizationErr{}
	}

	if euo, ok := a.children[0].op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != sigmoidOpType) {
		return a, noStabilizationErr{}
	}

	x := a.children[0].children[0]

	stabLogf("x : %v", x.Name())

	if retVal, err = Softplus(x); err == nil {
		retVal, err = Neg(retVal)
		if err != nil {
			return nil, errors.Wrap(err, negFail)
		}
		return retVal, nil
	}
	return nil, errors.Wrap(err, softplusFail)
}

/* Graph Optimizations */

// NegNegOptimization optimizes away -(-x) to just return x
// place before neg
func NegNegOptimization(a *Node) (retVal *Node, err error) {
	stabLogf("Optimizing -(-x)")
	enterLogScope()
	defer leaveLogScope()

	if euo, ok := a.op.(elemUnaryOp); !ok || (ok && euo.unaryOpType() != negOpType) {
		return a, noStabilizationErr{}
	}

	x := a.children[0]
	return x, nil
}
