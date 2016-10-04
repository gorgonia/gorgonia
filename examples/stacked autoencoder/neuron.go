package main

import . "github.com/chewxy/gorgonia"

type Neuron struct {
	w *Node
	b *Node

	g *ExprGraph
}

type initFn func() InitWFn

func NewNeuron(inputs, outputs, batchSize int, g *ExprGraph, fn initFn) *Neuron {
	w := NewMatrix(g, Float64, WithShape(inputs, outputs), WithInit(fn()))

	var b *Node
	if batchSize == 1 {
		b = NewVector(g, Float64, WithShape(outputs), WithInit(Zeroes()))
	} else {
		b = NewMatrix(g, Float64, WithShape(batchSize, outputs), WithInit(Zeroes()))
	}

	return &Neuron{
		w: w,
		b: b,
		g: g,
	}

}
