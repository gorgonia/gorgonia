package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type upsampleLayer struct {
	scale int
}

func (l *upsampleLayer) String() string {
	return fmt.Sprintf("Upsample layer: Scale->%[1]d", l.scale)
}

func (l *upsampleLayer) Type() string {
	return "upsample"
}

func (l *upsampleLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	upsampleOut, err := gorgonia.Upsample2D(input[0], l.scale)
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare upsample operation")
	}
	return upsampleOut, nil
}
