package main

import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
)

type ActivationFunction func(*Node) (*Node, error)

type Layer interface {
	Activate(x *Node) *Node
}

// LayerConsOpt is a option for constructing a layer
type LayerConsOpt func(l Layer)

func WithConf(inputs, outputs int) LayerConsOpt {
	f := func(lay Layer) {
		switch l := lay.(type) {
		case *DenoisingAutoencoder:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: 1}
		case *FC:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: 1}
		case *SoftmaxLayer:
			l.LayerConfig = LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: 1}
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

	g *ExprGraph

	af ActivationFunction
}

func NewFC(opts ...LayerConsOpt) *FC {
	fc := new(FC)
	fc.af = Sigmoid
	for _, opt := range opts {
		opt(fc)
	}

	u := func() InitWFn {
		high := 1 / float64(fc.Inputs)
		low := -high
		return Uniform(low, high)
	}

	fc.Neuron = NewNeuron(fc.Inputs, fc.Outputs, fc.g, u)
	return fc
}

func (l *FC) Activate(x *Node) *Node {
	xw := Must(Mul(x, l.w))
	defer func() {
		if r := recover(); r != nil {
			log.Printf("xw.shape: %v l.b.shape: %v", xw.Shape(), l.b.Shape())
			panic(r)
		}
	}()
	xwb := Must(Add(xw, l.b))
	act := Must(l.af(xwb))
	return act
}

type SoftmaxLayer struct {
	*Neuron
	LayerConfig
	g *ExprGraph
}

func NewSoftmaxLayer(opts ...LayerConsOpt) *SoftmaxLayer {
	sm := new(SoftmaxLayer)
	for _, opt := range opts {
		opt(sm)
	}

	u := func() InitWFn {
		high := 1 / float64(sm.Inputs)
		low := -high
		return Uniform(low, high)
	}

	sm.Neuron = NewNeuron(sm.Inputs, sm.Outputs, sm.g, u)
	return sm
}

func (l *SoftmaxLayer) Activate(x *Node) *Node {
	xw := Must(Mul(x, l.w))
	xwb := Must(Add(xw, l.b))
	act := Must(SoftMax(xwb))
	return act
}
