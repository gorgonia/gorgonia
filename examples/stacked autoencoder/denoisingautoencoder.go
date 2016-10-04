package main

import . "github.com/chewxy/gorgonia"

type DenoisingAutoencoder struct {
	*Neuron
	LayerConfig

	h  *Neuron
	af ActivationFunction

	input      *Node
	corruption *Node
	corrupted  *Node
	output     *Node
	g          *ExprGraph
}

func NewDA(corruption float64, opts ...LayerConsOpt) *DenoisingAutoencoder {
	da := new(DenoisingAutoencoder)
	da.af = Sigmoid
	for _, opt := range opts {
		opt(da)
	}

	u := func() InitWFn {
		return GlorotU(1.0)
	}

	da.Neuron = NewNeuron(da.Inputs, da.Outputs, da.BatchSize, da.g, u)
	da.h = new(Neuron)
	da.h.w = Must(Transpose(da.w))
	da.h.b = NewVector(da.g, Float64, WithShape(da.Outputs), WithInit(Zeroes()))

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
	if da.BatchSize == 1 {
		da.h.b = NewVector(da.g, Float64, WithShape(da.Inputs), WithInit(Zeroes()))
	} else {
		da.h.b = NewMatrix(da.g, Float64, WithShape(da.BatchSize, da.Inputs), WithInit(Zeroes()))
	}

	da.corruption = BinomialRandomNode(da.g, Float64, 1, corruption)

	return da
}

func (l *DenoisingAutoencoder) Activate() (retVal *Node, err error) {
	var xw, xwb *Node
	if xw, err = Mul(l.corrupted, l.w); err != nil {
		return
	}

	if xwb, err = Add(xw, l.b); err != nil {
		return
	}
	return l.af(xwb)
}

func (l *DenoisingAutoencoder) Reconstruct(y *Node) *Node {
	yw := Must(Mul(y, l.h.w))
	ywb := Must(Add(yw, l.h.b))
	act := Must(l.af(ywb))
	return act
}

func (l *DenoisingAutoencoder) Corrupt() (*Node, error) {
	return HadamardProd(l.corruption, l.input)
}

func (l *DenoisingAutoencoder) Cost(x *Node) (retVal *Node, err error) {
	if l.corrupted, err = l.Corrupt(); err != nil {
		return
	}
	var hidden *Node
	if hidden, err = l.Activate(); err != nil {
		return
	}

	y := l.Reconstruct(hidden)
	loss := Must(BinaryXent(l.input, y))
	return Mean(loss)
}
