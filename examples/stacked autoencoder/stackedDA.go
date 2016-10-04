package main

import (
	"fmt"
	"log"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/gorgonia/tensor/types"
)

type StackedDA struct {
	DeepConfig

	autoencoders []*DenoisingAutoencoder // ↓
	hiddenLayers []*FC                   // ↓
	final        *SoftmaxLayer           // ↓

	input *Node
	g     *ExprGraph
}

func NewStackedDA(g *ExprGraph, batchSize, size, inputs, outputs, layers int, hiddenSizes []int, corruptions []float64) *StackedDA {
	hiddenLayers := make([]*FC, layers)
	autoencoders := make([]*DenoisingAutoencoder, layers)

	var input, output *Node
	var err error
	if batchSize == 1 {
		input = NewVector(g, Float64, WithShape(inputs), WithName("x"))
	} else {
		input = NewMatrix(g, Float64, WithShape(batchSize, inputs), WithName("x"))
	}

	for i := 0; i < layers; i++ {
		var inputSize, outputSize int
		if i > 0 {
			outputSize = hiddenSizes[i]
			inputSize = hiddenSizes[i-1]
			input = output
		} else {
			inputSize = inputs
			outputSize = hiddenSizes[0]
		}

		hiddenLayers[i] = NewFC(WithGraph(g), WithConf(inputSize, outputSize, batchSize))
		hiddenLayers[i].input = input
		WithName(fmt.Sprintf("w_%d", i))(hiddenLayers[i].w)
		WithName(fmt.Sprintf("b_%d", i))(hiddenLayers[i].b)

		if output, err = hiddenLayers[i].Activate(); err != nil {
			panic(err)
		}

		autoencoders[i] = NewDATiedWeights(hiddenLayers[i].w, hiddenLayers[i].b, corruptions[i], WithGraph(g), WithConf(inputSize, outputSize, batchSize))
		autoencoders[i].input = input
	}

	final := NewSoftmaxLayer(WithGraph(g), WithConf(hiddenSizes[len(hiddenSizes)-1], outputs, batchSize))
	final.input = input
	WithName("Softmax_w")(final.w)
	WithName("Softmax_b")(final.b)

	conf := DeepConfig{
		LayerConfig:       LayerConfig{Inputs: inputs, Outputs: outputs, BatchSize: batchSize},
		Size:              size,
		Layers:            layers,
		HiddenLayersSizes: hiddenSizes,
	}

	return &StackedDA{
		DeepConfig: conf,

		autoencoders: autoencoders,
		hiddenLayers: hiddenLayers,
		final:        final,
		input:        input,
		g:            g,
	}
}

func (sda *StackedDA) Pretrain(x types.Tensor) (err error) {
	var inputs, model Nodes
	var machines []VM

	// logfile, err := os.OpenFile("exec.log", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	// logger := log.New(logfile, "", 0)

	inputs = Nodes{sda.input}
	for _, da := range sda.autoencoders {
		var cost *Node
		var grads Nodes
		cost, err = da.Cost(sda.input)
		// costs = append(costs, cost)

		// model = append(model, da.w, da.b, da.h.b)

		if grads, err = Grad(cost, da.w, da.b, da.h.b); err != nil {
			return
		}

		prog, locMap, err := CompileFunctionNEW(sda.g, inputs, grads)
		if err != nil {
			return err
		}

		log.Printf("%v, %d", prog, FmtNodeMap(locMap))
		// logger.SetPrefix(fmt.Sprintf("Train Layer %d:\t", i))
		// m := NewTapeMachine(prog, locMap, WithLogger(logger), WithWatchlist(), WithValueFmt("%+1.1s"))
		m := NewTapeMachine(prog, locMap)
		machines = append(machines, m)
	}

	solver := NewVanillaSolver()
	model = make(Nodes, 3)

	batches := x.Shape()[0] / sda.BatchSize
	var start int

	for batch := 0; batch < batches; batch++ {
		var input types.Tensor
		if input, err = tensor.Slice(x, S(start, start+sda.BatchSize)); err != nil {
			return
		}

		for i, da := range sda.autoencoders {
			model = model[:0]

			Let(sda.input, input)
			if err = machines[i].RunAll(); err != nil {
				return
			}
			model = append(model, da.w, da.b, da.h.b)
			solver.Step(model)
			machines[i].Reset()
		}
	}

	return nil
}

func (sda *StackedDA) Finetune(x types.Tensor, y []int) (err error) {
	var costs, model Nodes
	for i, da := range sda.autoencoders {
		if i > 0 {
			hidden := sda.hiddenLayers[i-1]
			if _, err = hidden.Activate(); err != nil {
				return
			}
		}

		model = append(model, da.w, da.b, da.h.b)
	}
	var probs *Node
	if probs, err = sda.final.Activate(); err != nil {
		return
	}

	logprobs := Must(Neg(Must(Log(probs))))
	for _, correct := range y {
		cost := Must(Slice(logprobs, S(correct)))
		costs = append(costs, cost)
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
