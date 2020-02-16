package gorgonia

import (
	"encoding/csv"
	"fmt"
	"io"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// This file deals with the compilation from a expression graph into a program
// that is executed by an interpreter

// Compile takes a graph and outputs a program suitable for *tapeMachine to run
func Compile(g *ExprGraph) (prog *program, locMap map[*Node]register, err error) {
	compileLogf("Compiling")
	enterLogScope()
	defer leaveLogScope()

	switch {
	case len(g.AllNodes()) == 0:
		err = errors.Errorf("Cannot compile an empty graph")
		return
	case g.Inputs().Len() == 0:
		err = errors.Errorf("Cannot compile a graph that has no input nodes")
		return
	}

	compileLogf("sorting")
	var sortedNodes Nodes
	if sortedNodes, err = Sort(g); err != nil {
		return nil, nil, errors.Wrap(err, sortFail)
	}
	reverseNodes(sortedNodes)

	df := analyze(g, sortedNodes)
	sortedNodes = df.insertDeviceInstr(sortedNodes)
	df.buildIntervals(sortedNodes)

	ra := newRegalloc(df)
	ra.alloc(sortedNodes)

	// debug related stuff
	df.debugIntervals(sortedNodes)
	logCompileState(g.name, g, df)

	inputs := g.Inputs()
	cg := newCodeGenerator(inputs, sortedNodes, df)
	prog, locMap = cg.gen()
	prog.cpulocs = ra.cpucount
	prog.gpulocs = ra.gpucount
	prog.cpumem = cg.cpumem
	prog.gpumem = cg.gpumem
	prog.df = df
	prog.g = g
	prog.sorted = sortedNodes

	return
}

// CompileFunction takes a graph, subsets it based on the input and output nodes provided and outputs a program suitable for *tapeMachine to run.
// It is analogous to theano.Function().
// If some input nodes are not used or is not reachable, this function will return an error
func CompileFunction(g *ExprGraph, inputs, outputs Nodes) (prog *program, locMap map[*Node]register, err error) {
	compileLogf("CompileFunctionNEW. Inputs: %d; outputs: %d", inputs, outputs)
	enterLogScope()
	defer leaveLogScope()

	subgraph := g.ExactSubgraphRoots(outputs...)
	var unused Nodes
	for _, in := range inputs {
		if !subgraph.all.Contains(in) {
			unused = append(unused, in)
		}
	}

	if len(unused) > 0 {
		return nil, nil, errors.Errorf("Not all the inputs are used: %v", unused)
	}

	var sortedNodes Nodes
	if sortedNodes, err = Sort(subgraph); err != nil {
		return nil, nil, errors.Wrap(err, sortFail)
	}
	reverseNodes(sortedNodes)

	df := analyze(subgraph, sortedNodes)
	sortedNodes = df.insertDeviceInstr(sortedNodes)
	df.buildIntervals(sortedNodes)

	ra := newRegalloc(df)
	ra.alloc(sortedNodes)

	cg := newCodeGenerator(inputs, sortedNodes, df)
	prog, locMap = cg.gen()
	prog.cpulocs = ra.cpucount
	prog.gpulocs = ra.gpucount
	prog.df = df
	prog.g = subgraph
	prog.sorted = sortedNodes

	return
}

// codgenerator holds the state for the code generation process
type codegenerator struct {
	locMap     map[*Node]register
	lastWrites map[register]*Node
	flushed    map[int]struct{}
	allocated  map[register]struct{}
	freed      map[register]struct{}
	deferFree  map[register]struct{}
	instrMap   map[*Node]fragment
	queue      []int // queue to flush

	lastReads map[register]int

	cpumem int64
	gpumem []int64

	g              *ExprGraph
	inputs, sorted Nodes
	df             *dataflow
	instructions   fragment
}

func newCodeGenerator(inputs, sorted Nodes, df *dataflow) *codegenerator {
	return &codegenerator{
		locMap:     make(map[*Node]register),
		lastWrites: make(map[register]*Node),
		flushed:    make(map[int]struct{}),
		allocated:  make(map[register]struct{}),
		freed:      make(map[register]struct{}),
		deferFree:  make(map[register]struct{}),
		instrMap:   make(map[*Node]fragment),
		lastReads:  make(map[register]int),

		g:      inputs[0].g,
		inputs: inputs,
		sorted: sorted,
		df:     df,
	}
}

// addInstr adds the instruction to the associated node in the instrMap.
// when we add instructions to the node map, we also try to determine the size of the allocations required
func (cg *codegenerator) addInstr(node *Node, instr tapeInstr) {
	if instrs := cg.instrMap[node]; instrs != nil {
		instrs = append(instrs, instr)
		cg.instrMap[node] = instrs
	} else {
		cg.instrMap[node] = fragment{instr}
	}

	var dt tensor.Dtype
	var err error
	switch inst := instr.(type) {
	case loadArg:
		if dt, err = dtypeOf(node.t); err != nil {
			panic(err)
		}
		d := instr.writes().device
		if d != CPU {
			if len(cg.gpumem) < int(d)+1 {
				diff := int(d) + 1 - len(cg.gpumem)
				cg.gpumem = append(cg.gpumem, make([]int64, diff)...)
			}
		}

		switch d {
		case CPU:
			cg.cpumem += calcMemSize(dt, node.Shape())
		default:
			cg.gpumem[int(d)] += calcMemSize(dt, node.Shape())
		}
	case alloc:
		if dt, err = dtypeOf(inst.t); err != nil {
			panic(err)
		}

		d := instr.writes().device
		if d != CPU {
			if len(cg.gpumem) < int(d)+1 {
				diff := int(d) + 1 - len(cg.gpumem)
				cg.gpumem = append(cg.gpumem, make([]int64, diff)...)
			}
		}

		switch d {
		case CPU:
			cg.cpumem += calcMemSize(dt, inst.s)
		default:
			cg.gpumem[int(d)] += calcMemSize(dt, inst.s)
		}
	case *execOp:
		if !inst.op.ReturnsPtr() {
			d := instr.writes().device
			if d != CPU {
				if len(cg.gpumem) < int(d)+1 {
					diff := int(d) + 1 - len(cg.gpumem)
					cg.gpumem = append(cg.gpumem, make([]int64, diff)...)
				}
			}
			switch d {
			case CPU:
				cg.cpumem += inst.size
			default:
				cg.gpumem[int(d)] += inst.size
			}
		}

	default:
		// panic("EHLP")
	}
}

// every time an instruction is added to the list of instructions,
// also add the instructionID and the register the instruction writes to.
// This helps with determining if a flushInstruction needs to be issued.
func (cg *codegenerator) updateLastWrites(reg register, n *Node) {
	cg.lastWrites[reg] = n
}

func (cg *codegenerator) flush() {
	compileLogf("Flushing")
	for _, instrID := range cg.queue {
		cg.flushed[instrID] = struct{}{}
	}
	cg.queue = cg.queue[:0]
}

func (cg *codegenerator) addArg(node *Node, interv *interval) {
	compileLogf("LoadArg: %x", node.ID())
	writeTo := interv.result

	cg.locMap[node] = writeTo
	instr := loadArg{
		// index:   index,
		index:   node.ID(),
		writeTo: writeTo,
		name:    node.Name(),
	}
	// cg.instructions = append(cg.instructions, instr)

	cg.addInstr(node, instr)
	cg.updateLastWrites(writeTo, node)
}

func (cg *codegenerator) addStmt(node *Node, interv *interval, i int) {
	compileLogf("Add Statement")
	enterLogScope()
	defer leaveLogScope()

	writeTo := interv.result

	var children Nodes
	var ok bool
	if children, ok = cg.df.devTransChildren[node]; !ok {
		children = node.children
	}

	switch op := node.op.(type) {
	case letOp:
		// there should be only 2 chilren
		if len(children) != 2 {
			panic("Expected only two children")
		}
		compileLogf("node.children %d. [1]: %v; [0]: %v", node.ID(), children[1], children[0])
		compileLogf("node isInput %v", node.isInput())
		from := cg.df.intervals[children[1]].result
		to := cg.df.intervals[children[0]].result

		instr := letInstr{
			readFrom: from,
			writeTo:  to,
		}
		// cg.instructions = append(cg.instructions, instr)

		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo, node)
	case readOp:
		// there should be only 1 child
		if len(children) != 1 {
			panic("Expected only one child")
		}
		compileLogf("node.children %d. [0]: %v", node.ID(), children[0])
		compileLogf("node isInput %v", node.isInput())
		compileLogf("node.children[0] Type %v, shape %v", children[0].t, children[0].shape)

		if _, ok := cg.flushed[i]; !ok {
			cg.addInstr(node, flushInstr{})
			cg.flush()
		}

		from := cg.df.intervals[children[0]].result
		instr := &readInstr{
			into:     op.into,
			readFrom: from,

			t: children[0].t,
			s: children[0].shape,
		}
		// cg.instructions = append(cg.instructions, instr)

		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo, node)
	case devTrans:
		if _, ok := cg.allocated[writeTo]; !ok {
			// insert new alloc
			var instr alloc
			instr = newAlloc(node, writeTo)
			// cg.instructions = append(cg.instructions, instr)

			cg.addInstr(node, instr)
			cg.updateLastWrites(writeTo, node)
			cg.queue = append(cg.queue, i)
			cg.allocated[writeTo] = struct{}{}
		}

		compileLogf("devTrans")
		if len(children) != 1 {
			panic("Expected only one child")
		}

		from := cg.df.intervals[children[0]].result
		to := cg.df.intervals[node].result

		instr := deviceTransport{
			from: from, to: to,
		}
		cg.addInstr(node, instr)

		if op.from != CPU && op.to == CPU {
			instrID := cg.sorted.index(op.toNode)
			if _, ok := cg.flushed[instrID]; !ok {
				// cg.instructions = append(cg.instructions, flushInstr{})
				cg.addInstr(node, flushInstr{})
				cg.flush()
			}
		}
		cg.updateLastWrites(writeTo, node)

	}
}

func (cg *codegenerator) addNode(node, replacement *Node, interv *interval, i int) {
	compileLogf("AddNode: %x %v", node.ID(), node.op)
	compileLogf("interval %v", interv)
	enterLogScope()
	defer leaveLogScope()

	writeTo := interv.result

	var reads []register
	var children Nodes
	var ok bool
	if children, ok = cg.df.devTransChildren[node]; !ok {
		children = node.children
	}
	for _, child := range children {
		cReplacement := cg.df.replacements[child]
		cInterv := cg.df.intervals[cReplacement]
		reads = append(reads, cInterv.result)
	}
	enterLogScope()
	defer leaveLogScope()

	var prealloc bool
	var useUnsafe bool
	// if it's not mutable, there is no chance it will be overwritten
	if node.isMutable() {
		// if the instruction calls an extern (cBLAS or cuBlas), then we should preallocate the vector
		if node.op.CallsExtern() {
			compileLogf("calls extern")
			if _, ok := cg.allocated[writeTo]; !ok {
				compileLogf("Inserting new alloc")
				var instr alloc
				instr = newAlloc(node, writeTo)
				// cg.instructions = append(cg.instructions, instr)

				cg.addInstr(node, instr)
				cg.updateLastWrites(writeTo, node)

				prealloc = true

				cg.queue = append(cg.queue, i)
				// cg.queue = append(cg.queue, len(cg.instructions)) // no -1.
				cg.allocated[writeTo] = struct{}{}
			}
		}
	}
	compileLogf("Node Reads %v", reads)
	// check if any previously buffered cBLAS or cuBLAS calls need to be flushed
	// it doesn't matter if the machine isn't using a batchedBLAS. flushInstr would just be a no-op at runtime
	for _, read := range reads {
		if lastWriteNode, ok := cg.lastWrites[read]; ok {
			instrID := cg.sorted.index(lastWriteNode)
			var op Op
			var onDev, nodeOnDev Device

			_, isDevTrans := lastWriteNode.Op().(devTrans)
			switch {
			case lastWriteNode.isArg(), lastWriteNode.isStmt && !isDevTrans:
				continue
			default:
				op = lastWriteNode.op
			}
			switch op.(type) {
			case CUDADoer:
				onDev = Device(0)
			case CLDoer:
				onDev = Device(0)
			default:
				onDev = CPU
			}

			switch node.op.(type) {
			case CUDADoer:
				nodeOnDev = Device(0)
			case CLDoer:
				nodeOnDev = Device(0)
			default:
				nodeOnDev = CPU
			}

			// if we have sequential Extern calls,  we just add it to the batch.
			// sequential in this can mean several instructions apart. For example:
			//		4 	A × B 	; read %2	; write to %3
			//		 	⋮	(doesn't use %3 or %10)
			//			⋮
			//		10  Aᵀ × B	; read %3	; write to %10
			//			⋮	(doesn't use %3, or %10)
			//			⋮
			//		12 	+		; read %10	; write to %12
			//
			// It is before instruction 12 that the flush will be added. 4 and 10 are considered sequential
			//
			// It is not sequential when both are not the same devices
			switch {
			case !op.CallsExtern():
				compileLogf("ToFlush: Node doesn't call extern. NO FLUSH")
				// op doesn't call extern... don't bother flushing
			case op.CallsExtern() && node.op.CallsExtern() && onDev == nodeOnDev && !isDevTrans:
				compileLogf("ToFlush: Both calls extern, both same device. NO FLUSH")
				// same device, both calls extern
				// no flush needed
			case op.CallsExtern() && node.op.CallsExtern() && onDev != nodeOnDev:
				compileLogf("ToFlush:  Differing devices")
				// different devices, both calls extern
				// flush needed
				fallthrough
			case op.CallsExtern() && !node.op.CallsExtern():
				compileLogf("ToFlush: Node requires value immediately")
				// node is gonna use the value immediately
				// flush needed
				fallthrough
			default:
				compileLogf("ToFlush: FLUSH")
				if _, ok := cg.flushed[instrID]; !ok {
					// cg.instructions = append(cg.instructions, flushInstr{})
					cg.addInstr(node, flushInstr{})
					cg.flush()
				}
			}

			// viaticum := cg.instructions[instrID] // ;) - it IS on the way
			// if instr, ok := viaticum.(*execOp); ok {
			// if op.CallsExtern() && !node.op.CallsExtern() {
			// }
			// }
		}

		// check the overwrites - if the overwrite and the resulting register is the same,
		// then use unsafe options when available
		overwrites := node.op.OverwritesInput()
		if overwrites >= 0 {
			compileLogf("Overwrites %d", overwrites)
			overwritten := reads[overwrites]
			compileLogf("NodeID:%d overwritten: %v, reads: %v, interval: %v", node.ID(), overwritten, interv.reads, interv.result)
			if overwritten == interv.result {
				compileLogf("Use unsafe")
				useUnsafe = true
			}
		}

	}

	cg.locMap[node] = writeTo

	// otherwise, the replacement has already been written
	if node == replacement {
		compileLogf("New Exec Op: %v", node.op)
		instr := newExecOp(node)
		instr.readFrom = reads
		instr.writeTo = writeTo
		instr.preAllocated = prealloc
		instr.useUnsafe = useUnsafe

		// cg.instructions = append(cg.instructions, instr)
		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo, node)
	}
}

func (cg *codegenerator) insertFree(instrID int, node *Node) {
	compileLogf("Inserting Free for instrID %d | instr: %v | op: %v", instrID, node, node.op)
	enterLogScope()
	defer leaveLogScope()

	var reads []register
	var children Nodes
	var ok bool
	if children, ok = cg.df.devTransChildren[node]; !ok {
		children = node.children
	}
	for _, child := range children {
		cReplacement := cg.df.replacements[child]
		cInterv := cg.df.intervals[cReplacement]
		reads = append(reads, cInterv.result)
	}
	compileLogf("reads %v", reads)

	// check if anything needs to be freed
	for _, read := range reads {
		var readNode *Node
		for n, reg := range cg.locMap {
			if reg == read {
				if readNode == nil {
					readNode = n
					continue
				}
				if readNode.id < n.id {
					readNode = n
				}
			}
		}
		// interv := cg.df.intervals[readNode]
		readRepl := cg.df.replacements[readNode]
		if readRepl == nil {
			readRepl = readNode
		}
		if readRepl == nil {
			continue
		}
		interv := cg.df.intervals[readRepl]
		compileLogf("interv for readRepl %v: %v", readRepl, interv)
		lastUse := interv.lastUse()
		compileLogf("Interval: %v; read: %v; Read Node %v; Op %v; LastUse %v; Instrid: %v", interv, read, readNode, readNode.op, lastUse, instrID)
		if lastUse >= 0 && lastUse <= instrID && read.device != CPU {
			if _, ok := cg.freed[read]; !ok {
				compileLogf("Adding Free %v. LastUse %d", read, interv.lastUse())
				// cg.instructions = append(cg.instructions, free{read})
				cg.addInstr(node, free{read})
				cg.freed[read] = struct{}{}
			}
		}
	}

	write := cg.locMap[node]
	repl := cg.df.replacements[node]
	interv := cg.df.intervals[repl]
	compileLogf("Node %v | write  %v | Last Use %v | %v", node, write, interv.lastUse(), node.isRoot())
	if interv.lastUse() == -1 || interv.lastUse() >= len(cg.sorted) {
		// if node.isRoot() {
		cg.deferFree[write] = struct{}{}
		// return
		// }

		// otherwise, it's essentially a NOOP, so we free the memory immediately after the Op is executed
		// TODO: do NO-OP optimizations
		// if _, ok := cg.freed[write]; !ok {
		// 	compileLogf("Adding Free %v. Last Use %d", write, interv.lastUse())
		// 	cg.addInstr(node, free{write})
		// 	cg.freed[write] = struct{}{}
		// }
	}
}

func (cg *codegenerator) insertLastFrees() int {
	node := cg.sorted[len(cg.sorted)-1]
	var instructionsAdded int
	for reg := range cg.deferFree {
		if _, ok := cg.freed[reg]; !ok {
			compileLogf("Adding Free %v to the final instruction", reg)
			cg.addInstr(node, free{reg})
			instructionsAdded++
		}
	}
	return instructionsAdded
}

func (cg *codegenerator) gen() (*program, map[*Node]register) {
	compileLogf("Generating from SORTED: %v", cg.sorted)
	enterLogScope()
	defer leaveLogScope()
	for i, node := range cg.sorted {
		// for i := len(cg.sorted) - 1; i ⩾ 0; i-- {
		// node := cg.sorted[i]
		replacement := cg.df.replacements[node]
		compileLogf("Working on %x. Replacement: %x", node.ID(), replacement.ID())

		nInterv := cg.df.intervals[replacement]
		switch {
		case node.isArg():
			cg.addArg(node, nInterv)
		case node.isStmt:
			cg.addStmt(node, nInterv, i)
		default:
			cg.addNode(node, replacement, nInterv, i)
		}
	}

	var instructionCount int
	for i := len(cg.sorted) - 1; i >= 0; i-- {
		node := cg.sorted[i]
		cg.insertFree(i, node)

		instructionCount += len(cg.instrMap[node])
	}

	instructionCount += cg.insertLastFrees()

	cg.instructions = make(fragment, 0, instructionCount)
	for _, node := range cg.sorted {
		instrs := cg.instrMap[node]
		cg.instructions = append(cg.instructions, instrs...)
	}

	return &program{
		instructions: cg.instructions,
		args:         len(cg.inputs),
		g:            cg.g,
		m:            cg.instrMap,
	}, cg.locMap
}

func compileState(w io.Writer, g *ExprGraph, df *dataflow) {
	header := []string{
		"ID", "Op", "Type", "Register", "Interval", "Used By", "Uses", "Overwrites", "BLAS?",
	}

	var rows [][]string
	for _, n := range g.AllNodes() {
		interv := df.intervals[n]

		row := make([]string, len(header))
		row[0] = fmt.Sprintf("%d", n.ID())
		row[2] = fmt.Sprintf("%s", n.t)
		row[3] = fmt.Sprintf("%s", interv.result)
		row[4] = fmt.Sprintf("%d - %d", interv.start, interv.end)
		row[5] = fmt.Sprintf("%v", interv.usePositions)
		row[6] = fmt.Sprintf("%d", n.children)

		if n.op != nil {
			row[1] = fmt.Sprintf("%s", n.op)
			overwrites := n.op.OverwritesInput()
			if overwrites >= 0 {
				row[7] = fmt.Sprintf("%d", n.children[overwrites].ID())
			}

			if n.op.CallsExtern() {
				row[8] = "yes"
			}
		}

		rows = append(rows, row)
	}
	cw := csv.NewWriter(w)
	cw.Comma = ';'
	// TODO: Check errors on writes here.
	cw.Write(header)
	cw.WriteAll(rows)
}
