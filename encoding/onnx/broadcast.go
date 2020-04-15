package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)

func broadcast(a, b *Node) (*gorgonia.Node, *gorgonia.Node, error) {
	return ggnBroadcast(a.gorgoniaNode, b.gorgoniaNode)
}

func ggnBroadcast(a, b *gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
	if sameDim(a, b) {
		return a, b, nil
	}
	// for NCHW tensors, the first dimension may be omited and must be broadcasted
	// TODO find a smarter way to achieve this
	switch {
	case len(a.Shape()) == 0:
		bDim := b.Shape()
		aRDim := make([]int, len(bDim))
		for i := 0; i < len(bDim); i++ {
			aRDim[i] = 1
		}
		aR, err := gorgonia.Reshape(a, aRDim)
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(aR, a, getBroadcastPattern(aR, b))
	case len(b.Shape()) == 0:
		aDim := a.Shape()
		bRDim := make([]int, len(aDim))
		for i := 0; i < len(aDim); i++ {
			bRDim[i] = 1
		}
		bR, err := gorgonia.Reshape(b, bRDim)
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(a, bR, getBroadcastPattern(a, bR))
	case len(a.Shape()) == 1 && len(b.Shape()) != 1:
		// Make an educated guess: find the axis that has the same dimension
		bShape := b.Shape()
		dims := make([]int, len(bShape))
		for i := 0; i < len(bShape); i++ {
			dims[i] = 1
			if bShape[i] == a.Shape()[0] {
				dims[i] = bShape[i]
			}
		}
		// Reshape node a
		aR, err := gorgonia.Reshape(a, dims)
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(aR, b, getBroadcastPattern(aR, b))
	case len(a.Shape()) != 1 && len(b.Shape()) == 1:
		// Make an educated guess: find the axis that has the same dimension
		aShape := a.Shape()
		dims := make([]int, len(aShape))
		for i := 0; i < len(aShape); i++ {
			dims[i] = 1
			if aShape[i] == b.Shape()[0] {
				dims[i] = aShape[i]
			}
		}
		// Reshape node a
		bR, err := gorgonia.Reshape(b, dims)
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(a, bR, getBroadcastPattern(a, bR))
	case len(a.Shape()) == 3 && len(b.Shape()) == 4:
		// Reshape node a
		aR, err := gorgonia.Reshape(a, append([]int{1}, a.Shape()...))
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(aR, b, getBroadcastPattern(aR, b))
	case len(a.Shape()) == 2 && len(b.Shape()) == 2:
		// Reshape node a
		return gorgonia.Broadcast(a, b, getBroadcastPattern(a, b))
	case len(a.Shape()) == 4 && len(b.Shape()) == 3:
		// Reshape node a
		bR, err := gorgonia.Reshape(b, append([]int{1}, b.Shape()...))
		if err != nil {
			return nil, nil, err
		}
		return gorgonia.Broadcast(a, bR, getBroadcastPattern(a, bR))
	default:
		return a, b, &onnx.ErrNotImplemented{
			Message: fmt.Sprintf("broadcast not yet implemented for shape %v, %v", a.Shape(), b.Shape()),
		}

	}
}

func sameDim(a, b *gorgonia.Node) bool {
	if len(a.Shape()) != len(b.Shape()) {
		return false
	}
	for i := 0; i < len(a.Shape()); i++ {
		if a.Shape()[i] != b.Shape()[i] {
			return false
		}
	}
	return true
}

func getBroadcastPattern(a, b *gorgonia.Node) gorgonia.BroadcastPattern {
	var leftAxes, rightAxes []byte
	for i := 0; i < len(a.Shape()); i++ {
		switch {
		case a.Shape()[i] == 1 && b.Shape()[i] != 1:
			leftAxes = append(leftAxes, byte(i))
		case a.Shape()[i] != 1 && b.Shape()[i] == 1:
			rightAxes = append(rightAxes, byte(i))
		}
	}
	return gorgonia.NewBroadcastPattern(leftAxes, rightAxes)

}
