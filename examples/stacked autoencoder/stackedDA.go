package main

import . "github.com/chewxy/gorgonia"

type StackedDA struct {
	DeepConfig

	autoencoders []*DenoisingAutoencoder // ↓
	hiddenLayers []*FC                   // ↓
	final        *SoftmaxLayer           // ↓

	g *ExprGraph
}

func NewStackedDA(g *ExprGraph, size, inputs, outputs, layers int, hiddenSizes []int, corruptions []float64) *StackedDA {
	hiddenLayers := make([]*FC, layers)
	autoencoders := make([]*DenoisingAutoencoder, layers)

	for i := 1; i < layers; i++ {
		outputSize := hiddenSizes[i]
		inputSize := hiddenSizes[i-1]

		hiddenLayers[i] = NewFC(WithGraph(g), WithConf(inputSize, outputSize))
		autoencoders[i] = NewDATiedWeights(hiddenLayers[i].w, hiddenLayers[i].b, corruptions[i], WithGraph(g), WithConf(inputSize, outputSize))
	}

	hiddenLayers[0] = NewFC(WithGraph(g), WithConf(inputs, hiddenSizes[0]))
	autoencoders[0] = NewDA(corruptions[0], WithGraph(g), WithConf(inputs, hiddenSizes[0]))

	conf := DeepConfig{
		LayerConfig:       LayerConfig{Inputs: inputs, Outputs: outputs},
		Size:              size,
		Layers:            layers,
		HiddenLayersSizes: hiddenSizes,
	}

	return &StackedDA{
		DeepConfig: conf,

		autoencoders: autoencoders,
		hiddenLayers: hiddenLayers,
		final:        NewSoftmaxLayer(WithGraph(g), WithConf(hiddenSizes[len(hiddenSizes)-1], outputs)),
		g:            g,
	}
}

func (sda *StackedDA) Pretrain(x *Node) (err error) {
	input := x
	var costs, model Nodes
	for i, da := range sda.autoencoders {
		if i > 0 {
			hidden := sda.hiddenLayers[i-1]
			input = hidden.Activate(input)
		}

		cost := da.Cost(input)
		costs = append(costs, cost)

		model = append(model, da.w, da.b, da.h.b)
	}

	g := sda.g.SubgraphRoots(costs...)
	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		return
	}

	solver := NewVanillaSolver()
	solver.Step(model)
	return nil
}

func (sda *StackedDA) Finetune(x, y *Node) {
	input := x

	var costs, model Nodes
	for i, da := range sda.autoencoders {
		if i > 0 {
			hidden := sda.hiddenLayers[i-1]
			input = hidden.Activate(input)
		}

		model = append(model, da.w, da.b, da.h.b)
	}
	probs := sda.last.Activate(input)
	loss := Must(BinaryXent(probs, y))
	cost := Must(Mean(loss))

	g := sda.g.SubgraphRoots(costs...)
	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		return
	}

	solver := NewVanillaSolver()
	solver.Step(model)
}
