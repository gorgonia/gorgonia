package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"runtime"
	"strconv"

	. "github.com/chewxy/gorgonia"
	"github.com/chewxy/gorgonia/tensor"
)

// model params
var hiddenSizes = []int{100}
var embeddingSize = 10

type layer struct {
	wix    *Node
	wih    *Node
	bias_i *Node

	wfx    *Node
	wfh    *Node
	bias_f *Node

	wox    *Node
	woh    *Node
	bias_o *Node

	wcx    *Node
	wch    *Node
	bias_c *Node
}

// single layer example
type model struct {
	g  *ExprGraph
	ls []*layer

	// decoder
	whd    *Node
	bias_d *Node

	embedding *Node

	inputVector *Node
	prevHiddens Nodes
	prevCells   Nodes

	prefix string
	free   bool
}

type lstmOut struct {
	hiddens Nodes
	cells   Nodes

	probs *Node
}

func NewLSTMModel(inputSize, embeddingSize, outputSize int, hiddenSizes []int) *model {
	m := new(model)
	g := NewGraph()

	log.Printf("EMB : %d HIDDEN SIZES: %v", embeddingSize, hiddenSizes)

	var hiddens, cells []*Node
	for depth := 0; depth < len(hiddenSizes); depth++ {
		prevSize := embeddingSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		l := new(layer)
		m.ls = append(m.ls, l) // add layer to model

		layerID := strconv.Itoa(depth)

		// input gate weights

		wixT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		wihT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		bias_iT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		l.wix = NewMatrix(g, Float32, WithName("wix_"+layerID), WithShape(hiddenSize, prevSize), WithValue(wixT))
		l.wih = NewMatrix(g, Float32, WithName("wih_"+layerID), WithShape(hiddenSize, hiddenSize), WithValue(wihT))
		l.bias_i = NewVector(g, Float32, WithName("bias_i_"+layerID), WithShape(hiddenSize), WithValue(bias_iT))

		// output gate weights

		woxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		wohT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		bias_oT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		l.wox = NewMatrix(g, Float32, WithName("wox_"+layerID), WithShape(hiddenSize, prevSize), WithValue(woxT))
		l.woh = NewMatrix(g, Float32, WithName("woh_"+layerID), WithShape(hiddenSize, hiddenSize), WithValue(wohT))
		l.bias_o = NewVector(g, Float32, WithName("bias_o_"+layerID), WithShape(hiddenSize), WithValue(bias_oT))

		// forget gate weights

		wfxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		wfhT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		bias_fT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		l.wfx = NewMatrix(g, Float32, WithName("wfx_"+layerID), WithShape(hiddenSize, prevSize), WithValue(wfxT))
		l.wfh = NewMatrix(g, Float32, WithName("wfh_"+layerID), WithShape(hiddenSize, hiddenSize), WithValue(wfhT))
		l.bias_f = NewVector(g, Float32, WithName("bias_f_"+layerID), WithShape(hiddenSize), WithValue(bias_fT))

		// cell write

		wcxT := tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		wchT := tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		bias_cT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		l.wcx = NewMatrix(g, Float32, WithName("wcx_"+layerID), WithShape(hiddenSize, prevSize), WithValue(wcxT))
		l.wch = NewMatrix(g, Float32, WithName("wch_"+layerID), WithShape(hiddenSize, hiddenSize), WithValue(wchT))
		l.bias_c = NewVector(g, Float32, WithName("bias_c_"+layerID), WithShape(hiddenSize), WithValue(bias_cT))

		// this is to simulate a default "previous" state
		hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		hidden := NewVector(g, Float32, WithName("prevHidden_"+layerID), WithShape(hiddenSize), WithValue(hiddenT))
		cell := NewVector(g, Float32, WithName("prevCell_"+layerID), WithShape(hiddenSize), WithValue(cellT))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}

	lastHiddenSize := hiddenSizes[len(hiddenSizes)-1]

	whdT := tensor.New(tensor.WithShape(outputSize, lastHiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, outputSize, lastHiddenSize)))
	bias_dT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(outputSize))

	m.whd = NewMatrix(g, Float32, WithName("whd_"), WithShape(outputSize, lastHiddenSize), WithValue(whdT))
	m.bias_d = NewVector(g, Float32, WithName("bias_d_"), WithShape(outputSize), WithValue(bias_dT))

	embeddingT := tensor.New(tensor.WithShape(inputSize, embeddingSize), tensor.WithBacking(Gaussian32(0.0, 0.008, inputSize, embeddingSize)))
	m.embedding = NewMatrix(g, Float32, WithName("embedding"), WithShape(inputSize, embeddingSize), WithValue(embeddingT))

	// these are to simulate a previous state
	dummyInputVec := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(embeddingSize)) // zeroes
	m.inputVector = NewVector(g, Float32, WithName("inputVector_"), WithShape(embeddingSize), WithValue(dummyInputVec))
	m.prevHiddens = hiddens
	m.prevCells = cells

	m.g = g
	return m
}

func (m *model) Clone() *model {
	m2 := new(model)
	m2.g = NewGraph()

	m2.ls = make([]*layer, len(m.ls))
	for i, l := range m.ls {
		m2.ls[i] = new(layer)

		m2.ls[i].wix = l.wix.CloneTo(m2.g)
		m2.ls[i].wih = l.wih.CloneTo(m2.g)
		m2.ls[i].bias_i = l.bias_i.CloneTo(m2.g)

		m2.ls[i].wfx = l.wfx.CloneTo(m2.g)
		m2.ls[i].wfh = l.wfh.CloneTo(m2.g)
		m2.ls[i].bias_f = l.bias_f.CloneTo(m2.g)

		m2.ls[i].wox = l.wox.CloneTo(m2.g)
		m2.ls[i].woh = l.woh.CloneTo(m2.g)
		m2.ls[i].bias_o = l.bias_o.CloneTo(m2.g)

		m2.ls[i].wcx = l.wcx.CloneTo(m2.g)
		m2.ls[i].wch = l.wch.CloneTo(m2.g)
		m2.ls[i].bias_c = l.bias_c.CloneTo(m2.g)
	}

	m2.whd = m.whd.CloneTo(m2.g)
	m2.bias_d = m.bias_d.CloneTo(m2.g)
	m2.embedding = m.embedding.CloneTo(m2.g)
	m2.inputVector = m.inputVector.CloneTo(m2.g)

	m2.prevHiddens = make(Nodes, len(m.prevHiddens))
	for i, h := range m.prevHiddens {
		m2.prevHiddens[i] = h.CloneTo(m2.g)
	}

	m2.prevCells = make(Nodes, len(m.prevCells))
	for i, c := range m.prevCells {
		m2.prevCells[i] = c.CloneTo(m2.g)
	}
	return m2
}

func (m *model) inputs() (retVal Nodes) {
	for _, l := range m.ls {
		lin := Nodes{
			l.wix,
			l.wih,
			l.bias_i,
			l.wfx,
			l.wfh,
			l.bias_f,
			l.wox,
			l.woh,
			l.bias_o,
			l.wcx,
			l.wch,
			l.bias_c,
		}

		retVal = append(retVal, lin...)
	}

	retVal = append(retVal, m.whd)
	retVal = append(retVal, m.bias_d)
	retVal = append(retVal, m.embedding)
	return
}

func (m *model) fwd(srcIndex int, prev *lstmOut) (retVal *lstmOut, err error) {
	// log.Printf("FORWARDING LSTM. SrcIndex: %v, prev: %v", srcIndex, prev == nil)
	var prevHiddens Nodes
	var prevCells Nodes

	if prev == nil {
		prevHiddens = m.prevHiddens
		prevCells = m.prevCells
	} else {
		prevHiddens = prev.hiddens
		prevCells = prev.cells
	}

	inputVector := m.inputVector

	var hiddens, cells Nodes
	for i, l := range m.ls {
		if i == 0 {
			inputVector = Must(Slice(m.embedding, S(srcIndex)))
		} else {
			inputVector = hiddens[i-1]
		}

		prevHidden := prevHiddens[i]
		prevCell := prevCells[i]

		var h0, h1, inputGate *Node
		h0 = Must(Mul(l.wix, inputVector))
		h1 = Must(Mul(l.wih, prevHidden))
		inputGate = Must(Sigmoid(Must(Add(Must(Add(h0, h1)), l.bias_i))))

		var h2, h3, forgetGate *Node
		h2 = Must(Mul(l.wfx, inputVector))
		h3 = Must(Mul(l.wfh, prevHidden))
		forgetGate = Must(Sigmoid(Must(Add(Must(Add(h2, h3)), l.bias_f))))

		var h4, h5, outputGate *Node
		h4 = Must(Mul(l.wox, inputVector))
		h5 = Must(Mul(l.woh, prevHidden))
		outputGate = Must(Sigmoid(Must(Add(Must(Add(h4, h5)), l.bias_o))))

		var h6, h7, cellWrite *Node
		h6 = Must(Mul(l.wcx, inputVector))
		h7 = Must(Mul(l.wch, prevHidden))
		cellWrite = Must(Tanh(Must(Add(Must(Add(h6, h7)), l.bias_c))))

		// cell activations
		var retain, write, cell, hidden *Node
		retain = Must(HadamardProd(forgetGate, prevCell))
		write = Must(HadamardProd(inputGate, cellWrite))
		cell = Must(Add(retain, write))
		hidden = Must(HadamardProd(outputGate, Must(Tanh(cell))))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}

	lastHidden := hiddens[len(hiddens)-1]
	var output *Node
	if output, err = Mul(m.whd, lastHidden); err == nil {
		if output, err = Add(output, m.bias_d); err != nil {
			WithName("LAST HIDDEN")(lastHidden)
			ioutil.WriteFile("err.dot", []byte(lastHidden.RestrictedToDot(3, 10)), 0644)
			panic(fmt.Sprintf("ERROR: %v", err))
		}
	}

	var probs *Node
	probs = Must(SoftMax(output))

	retVal = &lstmOut{
		hiddens: hiddens,
		cells:   cells,
		probs:   probs,
	}
	return
}

func (m *model) costFn(sentence string) (cost, perplexity *Node, n int, err error) {
	asRunes := []rune(sentence)
	n = len(asRunes)

	var prev *lstmOut
	var source, target rune

	// log.Printf("SENTENCE IS: %q, n: %d", sentence, n)

	for i := -1; i < n; i++ {
		if i == -1 {
			source = START
		} else {
			source = asRunes[i]
		}

		if i == n-1 {
			target = END
		} else {
			target = asRunes[i+1]
		}

		sourceId := vocabIndex[source]
		targetId := vocabIndex[target]

		var loss, perp *Node
		// cache

		if prev, err = m.fwd(sourceId, prev); err != nil {
			return
		}

		logprob := Must(Neg(Must(Log(prev.probs))))
		loss = Must(Slice(logprob, S(targetId)))
		log2prob := Must(Neg(Must(Log2(prev.probs))))
		perp = Must(Slice(log2prob, S(targetId)))

		if cost == nil {
			cost = loss
		} else {
			cost = Must(Add(cost, loss))
		}
		WithName("Cost")(cost)

		if perplexity == nil {
			perplexity = perp
		} else {
			perplexity = Must(Add(perplexity, perp))
		}
	}
	return
}

func (m *model) predict() {
	var sentence []rune
	var prev *lstmOut
	var err error

	for {
		var id int
		if len(sentence) > 0 {
			id = vocabIndex[sentence[len(sentence)-1]]
		}

		if prev, err = m.fwd(id, prev); err != nil {
			panic(err)
		}
		g := m.g.SubgraphRoots(prev.probs)
		// f, _ := os.Create("log1.log")
		// logger := log.New(f, "", 0)
		// machine := NewLispMachine(g, ExecuteFwdOnly(), WithLogger(logger), WithWatchlist(), LogBothDir())
		machine := NewLispMachine(g, ExecuteFwdOnly())
		machine.ForceCPU()
		if err := machine.RunAll(); err != nil {
			if ctxerr, ok := err.(contextualError); ok {
				ioutil.WriteFile("FAIL1.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)
			}
			log.Printf("ERROR1 while predicting with %p %+v", machine, err)
		}

		sampledID := sample(prev.probs.Value())
		var char rune // hur hur varchar
		if char = vocab[sampledID]; char == END {
			break
		}

		if len(sentence) > maxCharGen {
			break
		}

		sentence = append(sentence, char)
		// m.g.UnbindAllNonInputs()
	}

	var sentence2 []rune
	prev = nil
	for {
		var id int
		if len(sentence2) > 0 {
			id = vocabIndex[sentence2[len(sentence2)-1]]
		}

		if prev, err = m.fwd(id, prev); err != nil {
			panic(err)
		}

		g := m.g.SubgraphRoots(prev.probs)
		// f, _ := os.Create("log2.log")
		// logger := log.New(f, "", 0)
		// machine := NewLispMachine(g, ExecuteFwdOnly(), WithLogger(logger), WithWatchlist(), LogBothDir())
		machine := NewLispMachine(g, ExecuteFwdOnly())
		machine.ForceCPU()
		if err := machine.RunAll(); err != nil {
			if ctxerr, ok := err.(contextualError); ok {
				log.Printf("Instruction ID %v", ctxerr.InstructionID())
				ioutil.WriteFile("FAIL2.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)
			}
			log.Printf("ERROR2 while predicting with %p: %+v", machine, err)
		}

		sampledID := maxSample(prev.probs.Value())

		var char rune // hur hur varchar
		if char = vocab[sampledID]; char == END {
			break
		}

		if len(sentence2) > maxCharGen {
			break
		}

		sentence2 = append(sentence2, char)
	}
	m.g.UnbindAllNonInputs()

	fmt.Printf("Sampled: %q; \nArgMax: %q\n", string(sentence), string(sentence2))
}

func (m *model) run(iter int, solver Solver) (retCost, retPerp float32, err error) {
	defer runtime.GC()

	i := rand.Intn(len(sentences))
	sentence := sentences[i]

	var cost, perp *Node
	var n int

	cost, perp, n, err = m.costFn(sentence)
	if err != nil {
		return
	}

	var readCost *Node
	var readPerp *Node
	var costVal Value
	var perpVal Value

	var g *ExprGraph
	if iter%100 == 0 {
		readPerp = Read(perp, &perpVal)
		readCost = Read(cost, &costVal)
		g = m.g.SubgraphRoots(cost, readPerp, readCost)
	} else {
		g = m.g.SubgraphRoots(cost)
	}

	// f, _ := os.Create(fmt.Sprintf("FAIL%d.log", iter))
	// logger := log.New(f, "", 0)
	// machine := NewLispMachine(g, WithLogger(logger), WithValueFmt("%-1.1s"), LogBothDir(), WithWatchlist())
	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		if ctxerr, ok := err.(contextualError); ok {
			ioutil.WriteFile("FAIL.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)

		}
		return
	}

	err = solver.Step(m.inputs())
	if err != nil {
		return
	}

	if iter%100 == 0 {
		if sv, ok := perpVal.(Scalar); ok {
			v := sv.Data().(float32)
			retPerp = float32(math.Pow(2, float64(v)/(float64(n)-1)))
		}
		if cv, ok := costVal.(Scalar); ok {
			retCost = cv.Data().(float32)
		}
	}
	machine.UnbindAll() // here so that a reference to machine exists
	m.g.UnbindAllNonInputs()
	return
}
