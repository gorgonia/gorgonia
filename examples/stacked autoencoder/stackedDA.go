package main

import (
	"fmt"
	"log"
	"os"

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
	firstInput := input

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
		input:        firstInput,
		g:            g,
	}
}

func (sda *StackedDA) Pretrain(x types.Tensor, epoch int) (err error) {
	var inputs, model Nodes
	var machines []VM

	logfile, err := os.OpenFile("exec.log", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	logger := log.New(logfile, "", 0)

	inputs = Nodes{sda.input}
	var costValue Value
	for i, da := range sda.autoencoders {
		var cost *Node
		var grads Nodes
		cost, err = da.Cost(sda.input)
		Read(cost, &costValue)

		if grads, err = Grad(cost, da.w, da.b, da.h.b); err != nil {
			return
		}

		prog, locMap, err := CompileFunctionNEW(sda.g, inputs, grads)
		if err != nil {
			return err
		}
		if epoch == 0 {
			log.Printf("Layer: %d \n%v, %-#v", i, prog, FmtNodeMap(locMap))
		}
		// logger.SetPrefix(fmt.Sprintf("Train Layer %d:\t", i))
		var m VM
		if epoch == 0 {
			// m = NewTapeMachine(prog, locMap, WithLogger(logger), WithWatchlist(), WithValueFmt("%+1.1s"), WithInfWatch())
			m = NewTapeMachine(prog, locMap, WithLogger(logger), WithValueFmt("%+1.1s"), WithWatchlist(), TraceExec())
			// m = NewTapeMachine(prog, locMap, WithNaNWatch(), WithInfWatch())
		} else {
			m = NewTapeMachine(prog, locMap, WithNaNWatch(), WithInfWatch())
		}
		m = NewTapeMachine(prog, locMap)
		machines = append(machines, m)
	}

	solver := NewVanillaSolver(WithBatchSize(float64(sda.BatchSize)))
	model = make(Nodes, 3)

	batches := x.Shape()[0] / sda.BatchSize
	var start int

	for i, da := range sda.autoencoders {
		var layerCosts []float64
		for batch := 0; batch < batches; batch++ {
			var input types.Tensor
			if input, err = tensor.Slice(x, S(start, start+sda.BatchSize)); err != nil {
				return
			}

			// logfile.Truncate(0)
			// logfile.Seek(0, 0)
			// log.Printf("Layer %d", i)
			model = model[:0]
			Let(sda.input, input)
			if err = machines[i].RunAll(); err != nil {
				return
			}
			c := costValue.Data().(float64)
			layerCosts = append(layerCosts, c)
			model = append(model, da.w, da.b, da.h.b)

			solver.Step(model)
			machines[i].Reset()
		}
		log.Printf("Epoch: %d Avg Cost (Layer %d): %v", epoch, i, avgF64s(layerCosts))
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

	solver := NewVanillaSolver()
	for batch := 0; batch < sda.BatchSize; batch++ {
		start := batch * sda.BatchSize
		end := start + sda.BatchSize

		for i, correct := range y[start:end] {
			var cost *Node
			if cost, err = Slice(logprobs, S(i), S(correct)); err != nil {
				log.Printf("i %d, len(y): %d; %v; err: %v", i, len(y), logprobs.Shape(), err)
				return
			}
			costs = append(costs, cost)
		}

		g := sda.g.SubgraphRoots(costs...)
		machine := NewLispMachine(g)
		if err = machine.RunAll(); err != nil {
			return
		}

		solver.Step(model)

	}
	return nil
}
