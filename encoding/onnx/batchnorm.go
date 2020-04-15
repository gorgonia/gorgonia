package gorgonnx

import (
	"errors"

	"github.com/google/uuid"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

// SPEC: https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization
// Gorgonia implem: https://godoc.org/gorgonia.org/gorgonia#BatchNorm

type batchnorm struct {
	epsilon  float64
	momentum float64
}

func init() {
	register("BatchNormalization", newBatchNorm)
}
func newBatchNorm() operator {
	return &batchnorm{}
}

func (b *batchnorm) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, 5)
	if err != nil {
		return err
	}
	x, scaleN, biasN, meanN, varN := children[0].gorgoniaNode,
		children[1].gorgoniaNode,
		children[2].gorgoniaNode,
		children[3].gorgoniaNode,
		children[4].gorgoniaNode
	if len(x.Shape()) != 4 {
		return &onnx.ErrNotImplemented{
			Operator: "Batchnormalization",
			Message:  "Only CxBxHxW tensors are supported",
		}
	}
	batchNormOp := &fastBatchnorm{
		scale:   scaleN.Value(),
		bias:    biasN.Value(),
		mean:    meanN.Value(),
		varN:    varN.Value(),
		epsilon: float32(b.epsilon),
	}
	if x.Shape()[0] != 1 {
		// helper func
		apply := func(f func(a, b *gorgonia.Node) (*gorgonia.Node, error), a, b *gorgonia.Node) (*gorgonia.Node, error) {
			if len(b.Shape()) != 1 {
				return nil, errors.New("Batchnorm: wrong shape")
			}
			ba, err := gorgonia.Reshape(b, []int{1, b.Shape()[0], 1, 1})
			if err != nil {
				return nil, err
			}
			aa, bb, err := gorgonia.Broadcast(a, ba, gorgonia.NewBroadcastPattern(nil, []byte{0, 2, 3}))
			if err != nil {
				return nil, err
			}
			return f(aa, bb)
		}
		// xNorm = (x - meanN) / sqrt( varN + b.epsilon)
		// output = scaleN * xNorm + biasN
		xNorm1, err := apply(gorgonia.Sub, x, meanN)
		if err != nil {
			return err
		}
		epsilon := gorgonia.NewConstant(float32(b.epsilon), gorgonia.WithName("epsilon"+uuid.New().String()))
		xNorm21, err := gorgonia.Add(varN, epsilon)
		if err != nil {
			return err
		}
		xNorm2, err := gorgonia.Sqrt(xNorm21)
		if err != nil {
			return err
		}
		xNorm, err := apply(gorgonia.HadamardDiv, xNorm1, xNorm2)
		if err != nil {
			return err
		}
		output1, err := apply(gorgonia.HadamardProd, xNorm, scaleN)
		if err != nil {
			return err
		}
		n.gorgoniaNode, err = apply(gorgonia.Add, output1, biasN)
		return err
	}
	n.gorgoniaNode, err = gorgonia.ApplyOp(batchNormOp, x)
	return err

}

func (b *batchnorm) init(o onnx.Operation) error {
	b.epsilon = 1e-5
	b.momentum = 0.9
	if e, ok := o.Attributes["epsilon"]; ok {
		if v, ok := e.(float32); ok {
			b.epsilon = float64(v)
		} else {
			return errors.New("epsilon is not a float64")
		}
	}
	if e, ok := o.Attributes["momentum"]; ok {
		if v, ok := e.(float32); ok {
			b.momentum = float64(v)
		} else {
			return errors.New("momentum is not a float64")
		}
	}
	return nil
}
