package main

import (
	"gorgonia.org/gorgonia"
)

// YOLOv3 YOLOv3 architecture
type YOLOv3 struct {
	g                                 *gorgonia.ExprGraph
	classesNum, boxesPerCell, netSize int
	out                               []*gorgonia.Node
	layersInfo                        []string
}
