package main

import (
	"log"

	. "github.com/chewxy/gorgonia"
)

type DenoisingAutoencoder struct {
	Neuron
	LayerConfig

	h  Neuron
	af ActivationFunction

	input      *Node
	corruption *Node
	corrupted  *Node
	output     *Node
	hiddenOut  *Node
	g          *ExprGraph
}

func NewDATiedWeights(w, b *Node, corruption float64, opts ...LayerConsOpt) *DenoisingAutoencoder {
	da := new(DenoisingAutoencoder)
	da.af = Sigmoid

	for _, opt := range opts {
		opt(da)
	}

	da.w = w
	da.b = b

	da.h.w = Must(Transpose(w))
	log.Printf("w %v da.h.w %v", w.Shape(), da.h.w.Shape())
	if da.BatchSize == 1 {
		da.h.b = NewVector(da.g, dt, WithShape(da.Inputs), WithInit(Zeroes()))
	} else {
		da.h.b = NewMatrix(da.g, dt, WithShape(da.BatchSize, da.Inputs), WithInit(Zeroes()))
	}

	da.corruption = BinomialRandomNode(da.g, dt, 1, corruption)

	return da
}

func (l *DenoisingAutoencoder) Activate() (retVal *Node, err error) {
	if l.output != nil {
		return l.output, nil
	}

	var xw, xwb *Node
	if xw, err = Mul(l.corrupted, l.w); err != nil {
		return
	}

	if xwb, err = Add(xw, l.b); err != nil {
		return
	}

	if l.output, err = l.af(xwb); err != nil {
		return nil, err
	}
	return l.output, nil
}

func (l *DenoisingAutoencoder) Reconstruct() (err error) {
	if l.hiddenOut != nil {
		return nil
	}

	var yw, ywb *Node
	if yw, err = Mul(l.output, l.h.w); err != nil {
		return
	}

	if ywb, err = Add(yw, l.h.b); err != nil {
		return
	}

	if l.hiddenOut, err = l.af(ywb); err != nil {
		return
	}
	return nil
}

func (l *DenoisingAutoencoder) Corrupt() (err error) {
	if l.corrupted != nil {
		return nil
	}

	if l.corrupted, err = HadamardProd(l.corruption, l.input); err != nil {
		return
	}
	return nil
}

func (l *DenoisingAutoencoder) Cost(x *Node) (retVal *Node, err error) {
	if err = l.Corrupt(); err != nil {
		return
	}
	if _, err = l.Activate(); err != nil {
		return
	}
	if err = l.Reconstruct(); err != nil {
		return
	}

	var loss *Node
	if loss, err = BinaryXent(l.hiddenOut, l.input); err != nil {
		return
	}

	return Mean(loss)
}
