package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"strconv"

	. "github.com/chewxy/gorgonia"
	"gorgonia.org/tensor"
)

var hiddenSizes = []int{100}
var embeddingSize = 10

type layer struct {
	wix    Value
	wih    Value
	bias_i Value

	wfx    Value
	wfh    Value
	bias_f Value

	wox    Value
	woh    Value
	bias_o Value

	wcx    Value
	wch    Value
	bias_c Value
}

type lstm struct {
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

func newLSTMLayer(g *ExprGraph, l *layer, name string) *lstm {
	retVal := new(lstm)
	retVal.wix = NodeFromAny(g, l.wix, WithName("wix_"+name))
	retVal.wih = NodeFromAny(g, l.wih, WithName("wih_"+name))
	retVal.bias_i = NodeFromAny(g, l.bias_i, WithName("bias_i_"+name))

	retVal.wfx = NodeFromAny(g, l.wfx, WithName("wfx_"+name))
	retVal.wfh = NodeFromAny(g, l.wfh, WithName("wfh_"+name))
	retVal.bias_f = NodeFromAny(g, l.bias_f, WithName("bias_f_"+name))

	retVal.wox = NodeFromAny(g, l.wox, WithName("wox_"+name))
	retVal.woh = NodeFromAny(g, l.woh, WithName("woh_"+name))
	retVal.bias_o = NodeFromAny(g, l.bias_o, WithName("bias_o_"+name))

	retVal.wcx = NodeFromAny(g, l.wcx, WithName("wcx_"+name))
	retVal.wch = NodeFromAny(g, l.wch, WithName("wch_"+name))
	retVal.bias_c = NodeFromAny(g, l.bias_c, WithName("bias_c_"+name))
	return retVal
}

func (l *lstm) fwd(inputVector, prevHidden, prevCell *Node) (hidden, cell *Node) {
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
	var retain, write *Node
	retain = Must(HadamardProd(forgetGate, prevCell))
	write = Must(HadamardProd(inputGate, cellWrite))
	cell = Must(Add(retain, write))
	hidden = Must(HadamardProd(outputGate, Must(Tanh(cell))))
	return
}

// single layer example
type model struct {
	ls []*layer

	// decoder
	whd    Value
	bias_d Value

	embedding Value

	// metadata
	inputSize, embeddingSize, outputSize int
	hiddenSizes                          []int

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
	m.inputSize = inputSize
	m.embeddingSize = embeddingSize
	m.outputSize = outputSize
	m.hiddenSizes = hiddenSizes

	for depth := 0; depth < len(hiddenSizes); depth++ {
		prevSize := embeddingSize
		if depth > 0 {
			prevSize = hiddenSizes[depth-1]
		}
		hiddenSize := hiddenSizes[depth]
		l := new(layer)
		m.ls = append(m.ls, l) // add layer to model

		// input gate weights

		l.wix = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		l.wih = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		l.bias_i = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// output gate weights

		l.wox = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		l.woh = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		l.bias_o = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// forget gate weights

		l.wfx = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		l.wfh = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		l.bias_f = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))

		// cell write

		l.wcx = tensor.New(tensor.WithShape(hiddenSize, prevSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, prevSize)))
		l.wch = tensor.New(tensor.WithShape(hiddenSize, hiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, hiddenSize, hiddenSize)))
		l.bias_c = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
	}

	lastHiddenSize := hiddenSizes[len(hiddenSizes)-1]

	m.whd = tensor.New(tensor.WithShape(outputSize, lastHiddenSize), tensor.WithBacking(Gaussian32(0.0, 0.08, outputSize, lastHiddenSize)))
	m.bias_d = tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(outputSize))

	m.embedding = tensor.New(tensor.WithShape(inputSize, embeddingSize), tensor.WithBacking(Gaussian32(0.0, 0.008, inputSize, embeddingSize)))
	return m

}

type charRNN struct {
	*model
	g  *ExprGraph
	ls []*lstm

	// decoder
	whd    *Node
	bias_d *Node

	embedding *Node

	inputVector *Node
	prevHiddens Nodes
	prevCells   Nodes
}

func newCharRNN(m *model) *charRNN {
	r := new(charRNN)
	r.model = m
	g := NewGraph()
	r.g = g

	var hiddens, cells Nodes
	for depth := 0; depth < len(m.hiddenSizes); depth++ {
		hiddenSize := m.hiddenSizes[depth]
		layerID := strconv.Itoa(depth)
		l := newLSTMLayer(r.g, r.model.ls[depth], layerID)
		r.ls = append(r.ls, l)

		// this is to simulate a default "previous" state
		hiddenT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		cellT := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(hiddenSize))
		hidden := NewVector(g, Float32, WithName("prevHidden_"+layerID), WithShape(hiddenSize), WithValue(hiddenT))
		cell := NewVector(g, Float32, WithName("prevCell_"+layerID), WithShape(hiddenSize), WithValue(cellT))

		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}
	r.whd = NodeFromAny(r.g, m.whd, WithName("whd"))
	r.bias_d = NodeFromAny(r.g, m.bias_d, WithName("bias_d"))
	r.embedding = NodeFromAny(r.g, m.embedding, WithName("Embedding"))

	// these are to simulate a previous state
	dummyInputVec := tensor.New(tensor.Of(tensor.Float32), tensor.WithShape(r.embeddingSize)) // zeroes
	r.inputVector = NewVector(g, Float32, WithName("inputVector_"), WithShape(r.embeddingSize), WithValue(dummyInputVec))
	r.prevHiddens = hiddens
	r.prevCells = cells

	return r
}

func (r *charRNN) inputs() (retVal Nodes) {
	for _, l := range r.ls {
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

	retVal = append(retVal, r.whd)
	retVal = append(retVal, r.bias_d)
	retVal = append(retVal, r.embedding)
	return
}

func (r *charRNN) fwd(srcIndex int, prev *lstmOut) (retVal *lstmOut, err error) {
	prevHiddens := r.prevHiddens
	prevCells := r.prevCells
	if prev != nil {
		prevHiddens = prev.hiddens
		prevCells = prev.cells
	}

	inputVector := r.inputVector
	var hiddens, cells Nodes
	for i, l := range r.ls {
		if i == 0 {
			inputVector = Must(Slice(r.embedding, S(srcIndex)))
		} else {
			inputVector = hiddens[i-1]
		}
		prevHidden := prevHiddens[i]
		prevCell := prevCells[i]

		hidden, cell := l.fwd(inputVector, prevHidden, prevCell)
		hiddens = append(hiddens, hidden)
		cells = append(cells, cell)
	}
	lastHidden := hiddens[len(hiddens)-1]
	var output *Node
	if output, err = Mul(r.whd, lastHidden); err == nil {
		if output, err = Add(output, r.bias_d); err != nil {
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

func (r *charRNN) costFn(sentence string) (cost, perplexity *Node, n int, err error) {
	asRunes := []rune(sentence)
	n = len(asRunes)

	var prev *lstmOut
	var source, target rune
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

		if prev, err = r.fwd(sourceId, prev); err != nil {
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

func (r *charRNN) predict() {
	var sentence []rune
	var prev *lstmOut
	var err error

	for {
		var id int
		if len(sentence) > 0 {
			id = vocabIndex[sentence[len(sentence)-1]]
		}

		if prev, err = r.fwd(id, prev); err != nil {
			panic(err)
		}
		g := r.g.SubgraphRoots(prev.probs)
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
		// r.g.UnbindAllNonInputs()
	}

	var sentence2 []rune
	prev = nil
	for {
		var id int
		if len(sentence2) > 0 {
			id = vocabIndex[sentence2[len(sentence2)-1]]
		}

		if prev, err = r.fwd(id, prev); err != nil {
			panic(err)
		}

		g := r.g.SubgraphRoots(prev.probs)
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
	r.g.UnbindAllNonInputs()

	fmt.Printf("Sampled: %q; \nArgMax: %q\n", string(sentence), string(sentence2))
}

func (r *charRNN) cleanup() {
	r.g.UnbindAllNonInputs()
	for _, n := range r.g.AllNodes() {
		ReturnNode(n)
	}
}

func run(r *charRNN, iter int, solver Solver) (retCost, retPerp float32, err error) {
	i := rand.Intn(len(sentences))
	sentence := sentences[i]

	var cost, perp *Node
	var n int

	cost, perp, n, err = r.costFn(sentence)
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
		g = r.g.SubgraphRoots(cost, readPerp, readCost)
	} else {
		g = r.g.SubgraphRoots(cost)
	}

	// f, _ := os.Create("FAIL.log")
	// logger := log.New(f, "", 0)
	// machine := NewLispMachine(g, WithLogger(logger), WithValueFmt("%-1.1s"), LogBothDir(), WithWatchlist())

	machine := NewLispMachine(g)
	if err = machine.RunAll(); err != nil {
		if ctxerr, ok := err.(contextualError); ok {
			ioutil.WriteFile("FAIL.dot", []byte(ctxerr.Node().RestrictedToDot(3, 3)), 0644)

		}
		return
	}
	machine.UnbindAll()

	err = solver.Step(r.inputs())
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

	return
}
