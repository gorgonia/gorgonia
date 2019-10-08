package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TinyYOLOv2Net Tiniy YOLO v2 architecture
type TinyYOLOv2Net struct {
	g                                                                                                                                *gorgonia.ExprGraph
	classesNum, boxesPerCell                                                                                                         int
	convWeights0, convWeights2, convWeights4, convWeights6, convWeights8, convWeights10, convWeights12, convWeights13, convWeights14 *gorgonia.Node

	out *gorgonia.Node

	biases  map[string][]float32
	gammas  map[string][]float32
	means   map[string][]float32
	vars    map[string][]float32
	kernels map[string][]float32
}

// GetOutput Get last layer
func (tiny *TinyYOLOv2Net) GetOutput() *gorgonia.Node {
	return tiny.out
}

// NewTinyYOLOv2Net Constructor for TinyYOLOv2Net
func NewTinyYOLOv2Net(g *gorgonia.ExprGraph, classesNumber int, boxesPerCell int, weightsFile string) *TinyYOLOv2Net {
	data := ParseTinyYOLOv2(weightsFile)

	biases := map[string][]float32{
		"conv_0":  {},
		"conv_2":  {},
		"conv_4":  {},
		"conv_6":  {},
		"conv_8":  {},
		"conv_10": {},
		"conv_12": {},
		"conv_13": {},
		"conv_14": {},
	}
	gammas := map[string][]float32{
		"conv_0":  {},
		"conv_2":  {},
		"conv_4":  {},
		"conv_6":  {},
		"conv_8":  {},
		"conv_10": {},
		"conv_12": {},
		"conv_13": {},
		"conv_14": {},
	}
	means := map[string][]float32{
		"conv_0":  {},
		"conv_2":  {},
		"conv_4":  {},
		"conv_6":  {},
		"conv_8":  {},
		"conv_10": {},
		"conv_12": {},
		"conv_13": {},
		"conv_14": {},
	}
	vars := map[string][]float32{
		"conv_0":  {},
		"conv_2":  {},
		"conv_4":  {},
		"conv_6":  {},
		"conv_8":  {},
		"conv_10": {},
		"conv_12": {},
		"conv_13": {},
		"conv_14": {},
	}
	kernels := map[string][]float32{
		"conv_0":  {},
		"conv_2":  {},
		"conv_4":  {},
		"conv_6":  {},
		"conv_8":  {},
		"conv_10": {},
		"conv_12": {},
		"conv_13": {},
		"conv_14": {},
	}

	convShape0 := tensor.Shape([]int{16, 3, 3, 3})
	convShape2 := tensor.Shape([]int{32, 16, 3, 3})
	convShape4 := tensor.Shape([]int{64, 32, 3, 3})
	convShape6 := tensor.Shape([]int{128, 64, 3, 3})
	convShape8 := tensor.Shape([]int{256, 128, 3, 3})
	convShape10 := tensor.Shape([]int{512, 256, 3, 3})
	convShape12 := tensor.Shape([]int{1024, 512, 3, 3})
	convShape13 := tensor.Shape([]int{512, 1024, 3, 3})
	convShape14 := tensor.Shape([]int{(classesNumber + 5) * boxesPerCell, 512, 1, 1}) // 5 masks

	lastIdx := 5 // skip first 5 values
	epsilon := float32(0.000001)
	PrepareData(biases, gammas, means, vars, kernels, data, "conv_0", convShape0, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_0", convShape0, epsilon)
	convWeights0 := PrepareConv(g, convShape0, kernels["conv_0"], "conv_0")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_2", convShape2, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_2", convShape2, epsilon)
	convWeights2 := PrepareConv(g, convShape2, kernels["conv_2"], "conv_2")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_4", convShape4, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_4", convShape4, epsilon)
	convWeights4 := PrepareConv(g, convShape4, kernels["conv_4"], "conv_4")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_6", convShape6, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_6", convShape6, epsilon)
	convWeights6 := PrepareConv(g, convShape6, kernels["conv_6"], "conv_6")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_8", convShape8, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_8", convShape8, epsilon)
	convWeights8 := PrepareConv(g, convShape8, kernels["conv_8"], "conv_8")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_10", convShape10, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_10", convShape10, epsilon)
	convWeights10 := PrepareConv(g, convShape10, kernels["conv_10"], "conv_10")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_12", convShape12, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_12", convShape12, epsilon)
	convWeights12 := PrepareConv(g, convShape12, kernels["conv_12"], "conv_12")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_13", convShape13, &lastIdx, true, true)
	DenormalizeWeights(biases, gammas, means, vars, kernels, "conv_13", convShape13, epsilon)
	convWeights13 := PrepareConv(g, convShape13, kernels["conv_13"], "conv_13")

	PrepareData(biases, gammas, means, vars, kernels, data, "conv_14", convShape14, &lastIdx, false, true)
	convWeights14 := PrepareConv(g, convShape14, kernels["conv_14"], "conv_14")

	return &TinyYOLOv2Net{
		g:             g,
		classesNum:    classesNumber,
		boxesPerCell:  boxesPerCell,
		convWeights0:  convWeights0,
		convWeights2:  convWeights2,
		convWeights4:  convWeights4,
		convWeights6:  convWeights6,
		convWeights8:  convWeights8,
		convWeights10: convWeights10,
		convWeights12: convWeights12,
		convWeights13: convWeights13,
		convWeights14: convWeights14,
		biases:        biases,
		gammas:        gammas,
		means:         means,
		vars:          vars,
		kernels:       kernels,
	}
}

// PrepareConv Prepare convolutional kernels
func PrepareConv(g *gorgonia.ExprGraph, shape tensor.Shape, weights []float32, layerName string) *gorgonia.Node {
	customKernel := tensor.New(
		tensor.WithBacking(weights),
		tensor.WithShape(shape...),
	)
	kernels := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(shape...), gorgonia.WithName(layerName), gorgonia.WithValue(customKernel))
	return kernels
}

// PrepareBiases Biases preparations
func PrepareBiases(g *gorgonia.ExprGraph, shape tensor.Shape, biases map[string][]float32, layerName string, biasName string) *gorgonia.Node {
	iters := shape.TotalSize() / len(biases[layerName])
	newArr := []float32{}
	for i := 0; i < len(biases[layerName]); i++ {
		for j := 0; j < iters; j++ {
			newArr = append(newArr, biases[layerName][i])
		}
	}
	biasTensor := tensor.New(tensor.Of(tensor.Float32), tensor.WithBacking(newArr), tensor.WithShape(shape...))
	biasNode := gorgonia.NewTensor(g, tensor.Float32, 4, gorgonia.WithShape(shape...), gorgonia.WithValue(biasTensor), gorgonia.WithName(biasName))
	return biasNode
}
