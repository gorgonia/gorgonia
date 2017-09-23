package main

import "github.com/chewxy/gorgonia/tensor"

type LayerConfig struct {
	Inputs, Outputs int
	BatchSize       int
}

type DeepConfig struct {
	LayerConfig

	Size              int // total number of examples
	Layers            int
	HiddenLayersSizes []int
}

var dt tensor.Dtype = tensor.Float64
