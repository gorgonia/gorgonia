package main

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
