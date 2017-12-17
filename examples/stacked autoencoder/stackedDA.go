package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"os"

	. "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
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
	final.input = hiddenLayers[len(hiddenLayers)-1].output
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

func (sda *StackedDA) Pretrain(x tensor.Tensor, epoch int) (err error) {
	var inputs, model Nodes
	var machines []VM

	inputs = Nodes{sda.input}
	var costValue Value
	for _, da := range sda.autoencoders {
		var cost *Node
		var grads Nodes
		cost, err = da.Cost(sda.input)
		readCost := Read(cost, &costValue)

		if grads, err = Grad(cost, da.w, da.b, da.h.b); err != nil {
			return
		}

		outputs := make(Nodes, len(grads)+1)
		copy(outputs, grads)
		outputs[len(outputs)-1] = readCost

		prog, locMap, err := CompileFunction(sda.g, inputs, outputs)
		if err != nil {
			return err
		}

		var m VM
		m = NewTapeMachine(sda.g, WithPrecompiled(prog, locMap))
		machines = append(machines, m)
	}

	solver := NewVanillaSolver(WithBatchSize(float64(sda.BatchSize)))
	// solver := NewVanillaSolver()
	model = make(Nodes, 3)

	batches := x.Shape()[0] / sda.BatchSize
	avgCosts := make([]float64, len(sda.autoencoders))

	var start int
	for i, da := range sda.autoencoders {
		var layerCosts []float64
		for batch := 0; batch < batches; batch++ {
			var input tensor.Tensor
			if input, err = x.Slice(S(start, start+sda.BatchSize)); err != nil {
				return
			}

			model = model[:0]
			Let(sda.input, input)
			if err = machines[i].RunAll(); err != nil {
				return
			}
			c := costValue.Data().(float64)
			layerCosts = append(layerCosts, c)
			model = append(model, da.w, da.b, da.h.b)

			machines[i].Reset()
		}
		solver.Step(model)
		avgC := avgF64s(layerCosts)
		avgCosts[i] = avgC
	}

	// nicely format our costs
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "%d\t", epoch)
	for i, ac := range avgCosts {
		if i < len(avgCosts)-1 {
			fmt.Fprintf(&buf, "%v\t", ac)
		} else {
			fmt.Fprintf(&buf, "%v", ac)
		}
	}
	trainingLog.Println(buf.String())

	return nil
}

func (sda *StackedDA) Finetune(x tensor.Tensor, y []int, epoch int) (err error) {
	var model Nodes
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

	// solver := NewVanillaSolver(WithBatchSize(float64(sda.BatchSize)))
	solver := NewVanillaSolver()

	batches := x.Shape()[0] / sda.BatchSize

	var cost *Node
	bs := NewConstant(float64(sda.BatchSize))
	losses := make(Nodes, sda.BatchSize)
	losses = losses[:0]
	cvs := make([]float64, batches)
	cvs = cvs[:0]
	for batch := 0; batch < batches; batch++ {
		losses = losses[:0]
		start := batch * sda.BatchSize
		end := start + sda.BatchSize

		if start >= len(y) {
			break
		}

		for i, correct := range y[start:end] {
			var loss *Node

			if sda.BatchSize == 1 {
				if loss, err = Slice(logprobs, S(correct)); err != nil {
					return
				}
			} else {
				if loss, err = Slice(logprobs, S(i), S(correct)); err != nil {
					return
				}
			}

			losses = append(losses, loss)
		}
		// Manual way of meaning the costs: we first sum them up, then div by the batch size
		if cost, err = ReduceAdd(losses); err != nil {
			return
		}

		if cost, err = Div(cost, bs); err != nil {
			return
		}

		g := sda.g.SubgraphRoots(cost)

		var input tensor.Tensor
		if input, err = x.Slice(S(start, end)); err != nil {
			return
		}

		// machine := NewLispMachine(g, WithLogger(logger), LogBothDir(), WithWatchlist(), WithValueFmt("%+1.1s"))
		machine := NewLispMachine(g)
		Let(sda.input, input)
		if err = machine.RunAll(); err != nil {
			return
		}

		cvs = append(cvs, cost.Value().(Scalar).Data().(float64))
	}
	solver.Step(model)

	trainingLog.Printf("%d\t%v", epoch, avgF64s(cvs))
	return nil
}

func (sda *StackedDA) Forwards(x tensor.Tensor) (res tensor.Tensor, err error) {
	if sda.final.output == nil {
		panic("sda.final not set!")
	}

	probs := sda.final.output
	logprobs := Must(Neg(Must(Log(probs))))

	// subgraph, and create a machine
	g := sda.g.SubgraphRoots(logprobs)
	// machine := NewLispMachine(g, WithLogger(logger), LogBothDir(), WithWatchlist(), ExecuteFwdOnly(), WithValueFmt("%+s"))
	machine := NewLispMachine(g, ExecuteFwdOnly())

	Let(sda.input, x)
	if err = machine.RunAll(); err != nil {
		return
	}

	res = logprobs.Value().(tensor.Tensor)
	return
}

// Save saves the model
func (sda *StackedDA) Save(filename string) (err error) {
	var f io.WriteCloser
	if f, err = os.OpenFile(filename, os.O_TRUNC|os.O_CREATE|os.O_WRONLY, 0644); err != nil {
		return
	}

	encoder := gob.NewEncoder(f)
	for _, da := range sda.autoencoders {
		if err = encoder.Encode(da.Neuron); err != nil {
			return
		}

		if err = encoder.Encode(da.h); err != nil {
			return
		}
	}

	if err = encoder.Encode(sda.final.Neuron); err != nil {
		return
	}
	f.Close()
	return
}

func (sda *StackedDA) Load(filename string) (err error) {
	var f io.ReadCloser
	if f, err = os.Open(filename); err != nil {
		return
	}

	decoder := gob.NewDecoder(f)
	for _, da := range sda.autoencoders {
		if err = decoder.Decode(&da.Neuron); err != nil {
			return
		}

		if err = decoder.Decode(&da.h); err != nil {
			return
		}
	}

	if err = decoder.Decode(&sda.final.Neuron); err != nil {
		return
	}
	f.Close()
	return
}
