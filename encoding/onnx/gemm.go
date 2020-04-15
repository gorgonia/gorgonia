package gorgonnx

import (
	"errors"

	"github.com/google/uuid"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

func init() {
	register("Gemm", newGemm)
}

func newGemm() operator {
	return &gemm{
		alpha:  1.0,
		beta:   1.0,
		transA: false,
		transB: false,
	}
}

type gemm struct {
	alpha  float32 // Scalar multiplier for the product of input tensors A * B. default is 1.0
	beta   float32 // Scalar multiplier for input tensor C. default is 1.0
	transA bool    // Whether A should be transposed
	transB bool    // Whether B should be transposed
}

// Compute Y = alpha * A' * B' + beta * C, where
//  * input tensor A has shape (M, K) or (K, M),
//  * input tensor B has shape (K, N) or (N, K),
//  * input tensor C is broadcastable to shape (M, N),
//  * output tensor Y has shape (M, N).
// A will be transposed before doing the computation if attribute transA is non-zero,
// same for B and transB.
// This operator supports unidirectional broadcasting i
// (tensor C should be unidirectional broadcastable to tensor A * B);
//
// https://github.com/onnx/onnx/blob/master/docs/Operators.md#Gemm
func (m *gemm) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 3)
	if err != nil {
		return err
	}
	a := children[0].gorgoniaNode
	b := children[1].gorgoniaNode
	c := children[2].gorgoniaNode
	if m.transA {
		if a, err = gorgonia.Transpose(a); err != nil {
			return err
		}
	}
	if m.transB {
		if b, err = gorgonia.Transpose(b); err != nil {
			return err
		}
	}
	if len(a.Shape()) != 2 || len(b.Shape()) != 2 || len(c.Shape()) != 2 {
		return errors.New("gemm: input should be matrices")
	}
	//dimM := a.Shape()[0]
	//dimN := b.Shape()[1]
	ab, err := gorgonia.Mul(a, b)
	if err != nil {
		return err
	}
	alphaN := gorgonia.NewScalar(g.exprgraph, ab.Dtype(), gorgonia.WithValue(m.alpha), gorgonia.WithName("gemmAlpha"+uuid.New().String()))
	alphaB, abB, err := gorgonia.Broadcast(alphaN, ab, gorgonia.NewBroadcastPattern([]byte{0, 1}, nil))
	if err != nil {
		return err
	}
	alphaAB, err := gorgonia.HadamardProd(alphaB, abB)
	if err != nil {
		return err
	}
	betaN := gorgonia.NewScalar(g.exprgraph, ab.Dtype(), gorgonia.WithValue(m.beta), gorgonia.WithName("gemmbeta"+uuid.New().String()))
	betaB, cB, err := gorgonia.Broadcast(betaN, c, gorgonia.NewBroadcastPattern([]byte{0, 1}, nil))
	if err != nil {
		return err
	}
	betaC, err := gorgonia.HadamardProd(betaB, cB)
	if err != nil {
		return err
	}
	if alphaAB.Shape()[0] == betaC.Shape()[0] && alphaAB.Shape()[1] == betaC.Shape()[1] {
		n.gorgoniaNode, err = gorgonia.Add(alphaAB, betaC)
		if err != nil {
			return err
		}
	} else {
		alphaABB, betaCB, err := ggnBroadcast(alphaAB, betaC)
		if err != nil {
			return err
		}
		n.gorgoniaNode, err = gorgonia.Add(alphaABB, betaCB)
		if err != nil {
			return err
		}
	}
	return nil
}

func (m *gemm) init(o onnx.Operation) error {
	m.alpha = 1.0
	m.beta = 1.0
	if alpha, ok := o.Attributes["alpha"]; ok {
		if alpha, ok := alpha.(float32); ok {
			m.alpha = alpha
		} else {
			return errors.New("Gemm: alpha is not a float32")
		}
	}
	if beta, ok := o.Attributes["beta"]; ok {
		if beta, ok := beta.(float32); ok {
			m.beta = beta
		} else {
			return errors.New("Gemm: beta is not a float32")
		}
	}
	if transA, ok := o.Attributes["transA"]; ok {
		if transA, ok := transA.(int64); ok {
			if transA == 1 {
				m.transA = true
			}
		} else {
			return errors.New("Gemm: transA is not an int")
		}
	}
	if transB, ok := o.Attributes["transB"]; ok {
		if transB, ok := transB.(int64); ok {
			if transB == 1 {
				m.transB = true
			}
		} else {
			return errors.New("Gemm: transB is not an int")
		}
	}
	return nil
}
