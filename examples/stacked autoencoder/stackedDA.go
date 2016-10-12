package main

import (
	"bytes"
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

func (sda *StackedDA) Pretrain(x types.Tensor, epoch int) (err error) {
	var inputs, model Nodes
	var machines []VM

	inputs = Nodes{sda.input}
	var costValue Value
	for _, da := range sda.autoencoders {
		var cost *Node
		var grads Nodes
		cost, err = da.Cost(sda.input)
		Read(cost, &costValue)

		if grads, err = Grad(cost, da.w, da.b, da.h.b); err != nil {
			return
		}

		prog, locMap, err := CompileFunction(sda.g, inputs, grads)
		if err != nil {
			return err
		}

		var m VM
		m = NewTapeMachine(prog, locMap)
		machines = append(machines, m)
	}

	solver := NewVanillaSolver(WithBatchSize(float64(sda.BatchSize)))
	model = make(Nodes, 3)

	batches := x.Shape()[0] / sda.BatchSize
	avgCosts := make([]float64, len(sda.autoencoders))

	var start int
	for i, da := range sda.autoencoders {
		var layerCosts []float64
		for batch := 0; batch < batches; batch++ {
			var input types.Tensor
			if input, err = tensor.Slice(x, S(start, start+sda.BatchSize)); err != nil {
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

			solver.Step(model)
			machines[i].Reset()
		}
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

func (sda *StackedDA) Finetune(x types.Tensor, y []int, epoch int) (err error) {
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

	solver := NewVanillaSolver()

	costValues := make([]*Value, len(y))
	costValues = costValues[:0]

	batches := x.Shape()[0] / sda.BatchSize
	costs := make(Nodes, sda.BatchSize)
	for batch := 0; batch < batches; batch++ {
		costs = costs[:0]
		start := batch * sda.BatchSize
		end := start + sda.BatchSize

		if start >= len(y) {
			break
		}

		// logfile, _ := os.OpenFile(fmt.Sprintf("execlog/exec_%v_%v.log", epoch, batch), os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0644)
		// logger := log.New(logfile, "", 0)

		for i, correct := range y[start:end] {
			var cost *Node
			var costValue Value

			if sda.BatchSize == 1 {
				if cost, err = Slice(logprobs, S(correct)); err != nil {
					return
				}
			} else {
				if cost, err = Slice(logprobs, S(i), S(correct)); err != nil {
					return
				}
			}

			readCost := Read(cost, &costValue)

			costValues = append(costValues, &costValue)
			costs = append(costs, readCost)
		}

		g := sda.g.SubgraphRoots(costs...)

		var input types.Tensor
		if input, err = tensor.Slice(x, S(start, end)); err != nil {
			return
		}

		// machine := NewLispMachine(g, WithLogger(logger), LogBothDir(), WithWatchlist(), WithValueFmt("%+s"))
		machine := NewLispMachine(g)
		Let(sda.input, input)
		if err = machine.RunAll(); err != nil {
			return
		}

		solver.Step(model)
	}

	cvs := make([]float64, len(y))
	cvs = cvs[:0]
	for _, cv := range costValues {
		v := (*cv)
		if v == nil {
			continue
		}
		c := v.Data().(float64)
		cvs = append(cvs, c)
	}

	trainingLog.Printf("%d\t%v", epoch, avgF64s(cvs))
	return nil
}

func (sda *StackedDA) Forwards(x types.Tensor) (res types.Tensor, err error) {
	logfile, _ := os.OpenFile("exec.log", os.O_TRUNC|os.O_WRONLY|os.O_CREATE, 0644)
	logger := log.New(logfile, "", 0)

	for i := range sda.autoencoders {
		if i > 0 {
			hidden := sda.hiddenLayers[i-1]
			if _, err = hidden.Activate(); err != nil {
				return
			}
		}
	}
	var probs *Node
	if probs, err = sda.final.Activate(); err != nil {
		return
	}

	logprobs := Must(Neg(Must(Log(probs))))
	g := sda.g.SubgraphRoots(logprobs)
	machine := NewLispMachine(g, WithLogger(logger), LogBothDir(), WithWatchlist(), ExecuteFwdOnly(), WithValueFmt("%+s"))
	// machine := NewLispMachine(g, ExecuteFwdOnly())

	Let(sda.input, x)
	if err = machine.RunAll(); err != nil {
		return
	}

	res = logprobs.Value().(Tensor).Tensor

	return
}
