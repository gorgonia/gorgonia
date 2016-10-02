package main

import . "github.com/chewxy/gorgonia"

type Neuron struct {
	w *Node
	b *Node

	g *ExprGraph
}

type initFn func() InitWFn

func NewNeuron(inputs, outputs int, g *ExprGraph, fn initFn) *Neuron {
	w := NewMatrix(g, Float64, WithShape(inputs, outputs), WithInit(fn()))
	b := NewVector(g, Float64, WithShape(outputs), WithInit(Zeroes()))
	// w := NewVector(g, Float64, WithShape(outputs), WithInit(fn()), WithName("w"))
	// b := NewScalar(g, Float64, WithValue(0.0), WithName("b"))
	return &Neuron{
		w: w,
		b: b,
		g: g,
	}

}
