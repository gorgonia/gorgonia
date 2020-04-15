package gorgonnx

import (
	"encoding/binary"
	"errors"
	"hash"
	"hash/fnv"

	"github.com/chewxy/hm"
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type leakyRELU struct {
	alpha float32
}

func (l *leakyRELU) Arity() int {
	return 1
}

func (l *leakyRELU) Type() hm.Type {
	a := hm.TypeVariable('a')
	return hm.NewFnType(a, a)
}

func (l *leakyRELU) InferShape(inputs ...gorgonia.DimSizer) (tensor.Shape, error) {
	if inputs[0] == nil {
		return nil, errors.New("learyRELU: infershape failed, nil shape")
	}

	return inputs[0].(tensor.Shape), nil
}

func (l *leakyRELU) Do(inputs ...gorgonia.Value) (gorgonia.Value, error) {
	if len(inputs) != l.Arity() {
		return nil, errors.New("leakyrelu: wrong number of arguments")
	}
	t, ok := inputs[0].(*tensor.Dense)
	if !ok {
		return nil, errors.New("leakyrelu: only dense are supported")

	}
	switch t.Dtype() {
	case tensor.Float64:
		if vals, ok := t.Data().([]float64); ok {
			for i := range vals {
				if vals[i] >= 0 {
					continue
				}
				vals[i] *= float64(l.alpha)
			}
		} else {
			return nil, errors.New("expeced a []float64, but cannot cast")
		}
	case tensor.Float32:
		if vals, ok := t.Data().([]float32); ok {
			for i := range vals {
				if vals[i] >= 0 {
					continue
				}
				vals[i] *= l.alpha
			}
		} else {
			return nil, errors.New("expected a []float32, but cannot cast")
		}
	default:
		return nil, errors.New("LeakyRelu Unsupported type")
	}
	return t, nil
}

func (l *leakyRELU) ReturnsPtr() bool {
	return true
}

func (l *leakyRELU) CallsExtern() bool {
	return false
}

func (l *leakyRELU) OverwritesInput() int {
	return 0
}

func (l *leakyRELU) WriteHash(h hash.Hash) {
	if err := binary.Write(h, binary.LittleEndian, []byte(`learyRELU`)); err != nil {
		panic(err)
	}
}

func (l *leakyRELU) Hashcode() uint32 {
	h := fnv.New32a()
	l.WriteHash(h)
	return h.Sum32()
}

func (l *leakyRELU) String() string {
	return "leakyRELU"
}

func init() {
	register("LeakyRelu", newLeakyRELU)
}

func newLeakyRELU() operator {
	return &leakyRELU{}
}

func (l *leakyRELU) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	var nodes = make([]*gorgonia.Node, len(children))
	for i := 0; i < len(children); i++ {
		nodes[i] = children[i].gorgoniaNode
	}
	var err error
	n.gorgoniaNode, err = gorgonia.ApplyOp(l, nodes[0])
	//n.gorgoniaNode, err = gorgonia.LeakyRelu(nodes[0], float64(l.alpha))
	return err
}

func (l *leakyRELU) init(o onnx.Operation) error {
	l.alpha = 0.01
	if alpha, ok := o.Attributes["alpha"]; ok {
		if alpha, ok := alpha.(float32); ok {
			l.alpha = alpha
			return nil
		}
		return errors.New("alpha in not a float32")
	}
	return nil
}
