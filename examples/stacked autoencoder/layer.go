package main

import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
)

type ActivationFunction func(*Node) (*Node, error)

type Layer interface {
	Activate() (*Node, error)
}

// LayerConsOpt is a option for constructing a layer
type LayerConsOpt func(l Layer)

func WithConf(inputs, outputs, batchSize int) LayerConsOpt {
	f := func(lay Layer) {
		switch l := lay.(type) {
		case *DenoisingAutoencoder:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: batchSize}
		case *FC:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: batchSize}
		case *SoftmaxLayer:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: batchSize}
		default:
			panic(fmt.Sprintf("WithConf not implemented yet for %T", l))
		}
	}
	return f
}

func WithActivationFunction(af ActivationFunction) LayerConsOpt {
	f := func(lay Layer) {
		switch l := lay.(type) {
		case *DenoisingAutoencoder:
			l.af = af
		case *FC:
			l.af = af
		case *SoftmaxLayer:
		default:
			panic(fmt.Sprintf("WithActivationFunction not implemented for %T", l))
		}
	}
	return f
}

func WithGraph(g *ExprGraph) LayerConsOpt {
	f := func(lay Layer) {
		switch l := lay.(type) {
		case *DenoisingAutoencoder:
			l.g = g
		case *FC:
			l.g = g
		case *SoftmaxLayer:
			l.g = g
		}
	}
	return f
}

type FC struct {
	*Neuron
	LayerConfig

	af ActivationFunction

	input  *Node
	output *Node
	g      *ExprGraph
}

func NewFC(opts ...LayerConsOpt) *FC {
	fc := new(FC)
	fc.af = Sigmoid
	for _, opt := range opts {
		opt(fc)
	}

	u := func() InitWFn {
		return GlorotU(1.0)
	}
	fc.Neuron = NewNeuron(fc.Inputs, fc.Outputs, fc.BatchSize, fc.g, u)
	return fc
}

func (l *FC) Activate() (retVal *Node, err error) {
	log.Printf("l.input: %v, l.w: %v l.b %v", l.input.Shape(), l.w.Shape(), l.b.Shape())
	var xw, xwb *Node
	if xw, err = Mul(l.input, l.w); err != nil {
		return
	}

	if xwb, err = Add(xw, l.b); err != nil {
		return
	}

	if retVal, err = l.af(xwb); err != nil {
		return
	}

	l.output = retVal
	return
}

type SoftmaxLayer struct {
	*Neuron
	LayerConfig

	input  *Node
	output *Node
	g      *ExprGraph
}

func NewSoftmaxLayer(opts ...LayerConsOpt) *SoftmaxLayer {
	sm := new(SoftmaxLayer)
	for _, opt := range opts {
		opt(sm)
	}

	u := func() InitWFn {
		return GlorotU(1.0)
	}

	sm.Neuron = NewNeuron(sm.Inputs, sm.Outputs, sm.BatchSize, sm.g, u)
	return sm
}

func (l *SoftmaxLayer) Activate() (retVal *Node, err error) {
	var xw, xwb *Node
	if xw, err = Mul(l.input, l.w); err != nil {
		return
	}
	if xwb, err = Add(xw, l.b); err != nil {
		return
	}
	if retVal, err = SoftMax(xwb); err != nil {
		return
	}
	l.output = retVal
	return
}
