package tflite

import (
	"fmt"
	"strings"

	G "gorgonia.org/gorgonia"
)

// Export serializes the computation producing the output node as a TFLite
// model. input becomes the model input; every other leaf of the graph must
// have a bound Value — the trained weights — which is baked into the model
// as constants.
//
// The supported subset is the inference part of a multilayer perceptron:
// Mul (matrix multiplication), Add of a Mul result and a leaf (exported as
// the bias of a fully connected layer), Tanh, Sigmoid and SoftMax over the
// last axis.
func Export(output, input *G.Node) ([]byte, error) {
	e := &exporter{
		m:     NewModel(),
		input: input,
		memo:  map[*G.Node]Tensor{},
	}
	out, err := e.node(output)
	if err != nil {
		return nil, err
	}
	e.m.Output(out)
	return e.m.Bytes()
}

type exporter struct {
	m     *Model
	input *G.Node
	memo  map[*G.Node]Tensor
}

func children(n *G.Node) []*G.Node {
	it := n.Graph().From(n.ID())
	nodes := make([]*G.Node, 0, it.Len())
	for it.Next() {
		nodes = append(nodes, it.Node().(*G.Node))
	}
	return nodes
}

func isLeaf(n *G.Node) bool { return len(children(n)) == 0 }

func opType(n *G.Node) string { return fmt.Sprintf("%T", n.Op()) }

func isMatMul(n *G.Node) bool {
	if isLeaf(n) || opType(n) != "gorgonia.linAlgBinOp" {
		return false
	}
	s := n.Op().String()
	return strings.Contains(s, "×") && !strings.Contains(s, "ᵀ")
}

func isAdd(n *G.Node) bool {
	return !isLeaf(n) && opType(n) == "gorgonia.elemBinOp" && strings.HasPrefix(n.Op().String(), "+")
}

func isUnary(n *G.Node, name string) bool {
	return !isLeaf(n) && opType(n) == "gorgonia.elemUnaryOp" && n.Op().String() == name
}

func isSoftmax(n *G.Node) bool {
	if isLeaf(n) || opType(n) != "*gorgonia.softmaxOp" {
		return false
	}
	// Softmax{axis, isLog}: only plain softmax over the default axis maps
	// to the TFLite operator.
	s := n.Op().String()
	return (s == "Softmax{-1, false}()" || s == fmt.Sprintf("Softmax{%d, false}()", len(n.Shape())-1))
}

// values returns the leaf's bound value as float32s.
func values(n *G.Node) ([]float32, error) {
	v := n.Value()
	if v == nil {
		return nil, fmt.Errorf("%v: leaf other than the input must have a bound value", n.Name())
	}
	switch data := v.Data().(type) {
	case []float32:
		return data, nil
	case []float64:
		f := make([]float32, len(data))
		for i, x := range data {
			f[i] = float32(x)
		}
		return f, nil
	case float32:
		return []float32{data}, nil
	case float64:
		return []float32{float32(data)}, nil
	}
	return nil, fmt.Errorf("%v: unsupported value type %T", n.Name(), v.Data())
}

func (e *exporter) node(n *G.Node) (Tensor, error) {
	if t, ok := e.memo[n]; ok {
		return t, nil
	}
	t, err := e.convert(n)
	if err != nil {
		return t, err
	}
	e.memo[n] = t
	return t, nil
}

func (e *exporter) convert(n *G.Node) (Tensor, error) {
	switch {
	case n == e.input:
		return e.m.Input(name(n), n.Shape()), nil

	case isLeaf(n):
		data, err := values(n)
		if err != nil {
			return -1, err
		}
		return e.m.Constant(name(n), n.Shape(), data), nil

	case isAdd(n):
		// Add(Mul(x, w), b) is a fully connected layer with a bias.
		kids := children(n)
		mm, bias := kids[0], kids[1]
		if !isMatMul(mm) {
			mm, bias = bias, mm
		}
		if !isMatMul(mm) || !isLeaf(bias) {
			return -1, fmt.Errorf("%v: only Add(Mul(x, w), bias) is supported", n.Op())
		}
		return e.fullyConnected(mm, bias)

	case isMatMul(n):
		return e.fullyConnected(n, nil)

	case isUnary(n, "tanh"):
		x, err := e.node(children(n)[0])
		if err != nil {
			return -1, err
		}
		return e.m.Tanh(x), nil

	case isUnary(n, "sigmoid"):
		x, err := e.node(children(n)[0])
		if err != nil {
			return -1, err
		}
		return e.m.Logistic(x), nil

	case isSoftmax(n):
		x, err := e.node(children(n)[0])
		if err != nil {
			return -1, err
		}
		return e.m.Softmax(x, 1.0), nil
	}
	return -1, fmt.Errorf("unsupported op %v (%v)", n.Op(), opType(n))
}

// fullyConnected exports Mul(x, w) with an optional bias leaf as a TFLite
// FULLY_CONNECTED op. gorgonia weights are [in, out]; the TFLite filter is
// [out, in].
func (e *exporter) fullyConnected(mm, bias *G.Node) (Tensor, error) {
	kids := children(mm)
	xNode, wNode := kids[0], kids[1]
	if !isLeaf(wNode) {
		return -1, fmt.Errorf("%v: the second operand of Mul must be a weight leaf", mm.Op())
	}
	x, err := e.node(xNode)
	if err != nil {
		return -1, err
	}

	ws := wNode.Shape()
	if len(ws) != 2 {
		return -1, fmt.Errorf("%v: weights must be a matrix, got %v", wNode.Name(), ws)
	}
	in, out := ws[0], ws[1]
	w, err := values(wNode)
	if err != nil {
		return -1, err
	}
	filter := make([]float32, len(w))
	for i := 0; i < in; i++ {
		for j := 0; j < out; j++ {
			filter[j*in+i] = w[i*out+j]
		}
	}
	f := e.m.Constant(name(wNode), []int{out, in}, filter)

	b := make([]float32, out)
	biasName := name(wNode) + "/zero_bias"
	if bias != nil {
		if b, err = values(bias); err != nil {
			return -1, err
		}
		if len(b) != out {
			return -1, fmt.Errorf("%v: bias has %d values, want %d", bias.Name(), len(b), out)
		}
		biasName = name(bias)
	}
	bt := e.m.Constant(biasName, []int{out}, b)

	return e.m.FullyConnected(x, f, bt, ActNone), nil
}

func name(n *G.Node) string {
	if s := n.Name(); s != "" {
		return s
	}
	return fmt.Sprintf("node_%d", n.ID())
}
