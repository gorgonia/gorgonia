package main

import . "github.com/chewxy/gorgonia"

type DenoisingAutoencoder struct {
	*Neuron
	LayerConfig

	h          *Neuron
	corruption *Node
	af         ActivationFunction
	g          *ExprGraph
}

func NewDA(corruption float64, opts ...LayerConsOpt) *DenoisingAutoencoder {
	da := new(DenoisingAutoencoder)
	da.af = Sigmoid
	for _, opt := range opts {
		opt(da)
	}
	u := func() InitWFn {
		high := 1 / float64(da.Inputs)
		low := -high
		return Uniform(low, high)
	}

	da.Neuron = NewNeuron(da.Inputs, da.Outputs, da.g, u)
	da.h = new(Neuron)
	da.h.w = Must(Transpose(da.w))
	da.h.b = NewVector(da.g, Float64, WithShape(da.Inputs), WithInit(Zeroes()))

	da.corruption = BinomialRandomNode(da.g, Float64, 1, corruption)

	return da
}

func NewDATiedWeights(w, b *Node, corruption float64, opts ...LayerConsOpt) *DenoisingAutoencoder {
	da := new(DenoisingAutoencoder)
	da.af = Sigmoid

	for _, opt := range opts {
		opt(da)
	}

	da.Neuron = new(Neuron)
	da.h = new(Neuron)
	da.w = w
	da.b = b

	da.h.w = Must(Transpose(da.w))
	da.h.b = NewVector(da.g, Float64, WithShape(da.Inputs), WithInit(Zeroes()))

	da.corruption = BinomialRandomNode(da.g, Float64, 1, corruption)

	return da
}

func (l *DenoisingAutoencoder) Activate(x *Node) *Node {
	xw := Must(Mul(x, l.h.w))
	xwb := Must(Add(xw, l.h.b))
	act := Must(l.af(xwb))
	return act
}

func (l *DenoisingAutoencoder) Reconstruct(y *Node) *Node {
	yw := Must(Mul(y, l.w))
	ywb := Must(Add(yw, l.b))
	act := Must(l.af(ywb))
	return act
}

func (l *DenoisingAutoencoder) Corrupt(x *Node) *Node {
	return Must(HadamardProd(l.corruption, x))
}

func (l *DenoisingAutoencoder) Cost(x *Node) *Node {
	corrupted := l.Corrupt(x)
	hidden := l.Activate(corrupted)
	y := l.Reconstruct(hidden)

	loss := Must(BinaryXent(x, y))
	return Must(Mean(loss))
}
