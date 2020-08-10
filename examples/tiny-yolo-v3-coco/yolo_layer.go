package main

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
)

type yoloLayer struct {
	masks          []int
	anchors        [][2]int
	flattenAnchors []int
	inputSize      int
	classesNum     int
	ignoreThresh   float32
}

func (l *yoloLayer) String() string {
	str := "YOLO layer: "
	for m := range l.masks {
		str += fmt.Sprintf("Mask->%[1]d Anchors->[%[2]d, %[3]d]", l.masks[m], l.anchors[m][0], l.anchors[m][1])
		if m != len(l.masks)-1 {
			str += "\t|\t"
		}
	}
	return str
}

func (l *yoloLayer) Type() string {
	return "yolo"
}

func (l *yoloLayer) ToNode(g *gorgonia.ExprGraph, input ...*gorgonia.Node) (*gorgonia.Node, error) {
	inputN := input[0]
	if len(inputN.Shape()) == 0 {
		return nil, fmt.Errorf("Input shape for YOLO layer is nil")
	}
	flattenAnchorsF32 := make([]float32, len(l.flattenAnchors))
	for i := range flattenAnchorsF32 {
		flattenAnchorsF32[i] = float32(l.flattenAnchors[i])
	}
	masksIdx := make([]int, len(l.masks))
	for i := range l.masks {
		masksIdx[i] = i
	}
	yoloNode, err := gorgonia.YOLOv3(inputN, flattenAnchorsF32, masksIdx, l.inputSize, l.classesNum, l.ignoreThresh)
	if err != nil {
		return nil, errors.Wrap(err, "Can't prepare YOLOv3 operation")
	}
	return yoloNode, nil
}
