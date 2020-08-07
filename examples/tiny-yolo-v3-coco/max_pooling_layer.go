package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type maxPoolingLayer struct {
	size   int
	stride int
}

func (l *maxPoolingLayer) String() string {
	return fmt.Sprintf("Maxpooling layer: Size->%[1]d Stride->%[2]d", l.size, l.stride)
}

func (l *maxPoolingLayer) Type() string {
	return "maxpool"
}

func (l *maxPoolingLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	shp := input[0].Shape()
	if shp[2]%2 == 0 {
		maxpoolOut, err := gorgonia.MaxPool2D(input[0], tensor.Shape{l.size, l.size}, []int{0, 0}, []int{l.stride, l.stride})
		if err != nil {
			return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare max pooling operation")
		}
		return maxpoolOut, nil
	}
	maxpoolOut, err := gorgonia.MaxPool2D(input[0], tensor.Shape{l.size, l.size}, []int{0, 1, 0, 1}, []int{l.stride, l.stride})
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare max pooling operation")
	}
	return maxpoolOut, nil
}
