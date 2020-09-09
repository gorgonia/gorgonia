package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type routeLayer struct {
	firstLayerIdx  int
	secondLayerIdx int
}

func (l *routeLayer) String() string {
	if l.secondLayerIdx != -1 {
		return fmt.Sprintf("Route layer: Start->%[1]d End->%[2]d", l.firstLayerIdx, l.secondLayerIdx)
	}
	return fmt.Sprintf("Route layer: Start->%[1]d", l.firstLayerIdx)
}

func (l *routeLayer) Type() string {
	return "route"
}

func (l *routeLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	concatNodes := []*gorgonia.Node{}
	concatNodes = append(concatNodes, input[l.firstLayerIdx])
	if l.secondLayerIdx > 0 {
		concatNodes = append(concatNodes, input[l.secondLayerIdx])
	}
	routeNode, err := gorgonia.Concat(1, concatNodes...)
	if err != nil {
		return &gorgonia.Node{}, errors.Wrap(err, "Can't prepare route operation")
	}

	return routeNode, nil
}
