package main

import (
	"fmt"

	"gorgonia.org/gorgonia"
)

// YOLOv3 YOLOv3 architecture
type YOLOv3 struct {
	g                                 *gorgonia.ExprGraph
	classesNum, boxesPerCell, netSize int
	out                               []*gorgonia.Node
	layersInfo                        []string
}

// Print Print architecture of network
func (net *YOLOv3) Print() {
	for i := range net.layersInfo {
		fmt.Println(net.layersInfo[i])
	}
}

// GetOutput Get out YOLO layers (can be multiple of them)
func (net *YOLOv3) GetOutput() []*gorgonia.Node {
	return net.out
}
