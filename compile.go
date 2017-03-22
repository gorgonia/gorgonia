package gorgonia

import (
	"encoding/csv"
	"fmt"
	"io"

	"github.com/pkg/errors"
)

// This file deals with the compilation from a expression graph into a program
// that is executed by an interpreter

// Compile takes a graph and outputs a program suitable for *tapeMachine to run
func Compile(g *ExprGraph) (prog *program, locMap map[*Node]register, err error) {
	compileLogf("Compiling")
	enterLoggingContext()
	defer leaveLoggingContext()

	compileLogf("sorting")
	var sortedNodes Nodes
	if sortedNodes, err = Sort(g); err != nil {
		return nil, nil, errors.Wrap(err, sortFail)
	}

	inputs := g.Inputs()

	df := analyze(g, sortedNodes)
	df.intervals = buildIntervals(sortedNodes)

	ra := newRegalloc(df)
	ra.alloc(sortedNodes)

	compileLogf("Intervals: %+v", FmtNodeMap(df.intervals))
	logCompileState(g.name, g, df)

	cg := newCodeGenerator(inputs, sortedNodes, df)
	prog, locMap = cg.gen()
	// prog, locMap = codegen(inputs, sortedNodes, df)
	prog.locs = ra.count
	prog.df = df
	prog.g = g
	prog.sorted = sortedNodes

	return
}

// CompileFunction takes a graph, subsets it based on the input and output nodes provided and outputs a program suitable for *tapeMachine to run.
// It is analogous to theano.Function().
func CompileFunction(g *ExprGraph, inputs, outputs Nodes) (prog *program, locMap map[*Node]register, err error) {
	compileLogf("CompileFunctionNEW. Inputs: %d; outputs: %d", inputs, outputs)
	enterLoggingContext()
	defer leaveLoggingContext()

	subgraph := g.SubgraphRoots(outputs...)
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

	df := analyze(subgraph, sortedNodes)
	df.intervals = buildIntervals(sortedNodes)

	ra := newRegalloc(df)
	ra.alloc(sortedNodes)

	cg := newCodeGenerator(inputs, sortedNodes, df)
	prog, locMap = cg.gen()
	// prog, locMap = codegen(inputs, sortedNodes, df)
	prog.locs = ra.count
	prog.df = df
	prog.g = subgraph
	prog.sorted = sortedNodes

	return
}

// codgenerator holds the state for the code generation process
type codegenerator struct {
	locMap     map[*Node]register
	lastWrites map[int]int
	flushed    map[int]struct{}
	instrMap   map[*Node]fragment
	queue      []int // queue to flus

	g              *ExprGraph
	inputs, sorted Nodes
	df             *dataflow
	instructions   fragment
}

func newCodeGenerator(inputs, sorted Nodes, df *dataflow) *codegenerator {
	return &codegenerator{
		locMap:     make(map[*Node]register),
		lastWrites: make(map[int]int),
		flushed:    make(map[int]struct{}),
		instrMap:   make(map[*Node]fragment),

		g:      inputs[0].g,
		inputs: inputs,
		sorted: sorted,
		df:     df,
	}
}

func (cg *codegenerator) addInstr(node *Node, instr tapeInstr) {
	if instrs := cg.instrMap[node]; instrs != nil {
		instrs = append(instrs, instr)
		cg.instrMap[node] = instrs
	} else {
		cg.instrMap[node] = fragment{instr}
	}
}

// every time an instruction is added to the list of instructions,
// also add the instructionID and the register the instruction writes to.
// This helps with determining if a flushInstruction needs to be issued.
func (cg *codegenerator) updateLastWrites(id int) {
	cg.lastWrites[id] = len(cg.instructions) - 1
}

func (cg *codegenerator) flush() {
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
	}
	cg.instructions = append(cg.instructions, instr)

	cg.addInstr(node, instr)
	cg.updateLastWrites(writeTo.id)
}

func (cg *codegenerator) addStmt(node *Node, interv *interval) {
	compileLogf("Statement")
	writeTo := interv.result

	switch op := node.op.(type) {
	case letOp:
		// there should be only 2 chilren
		if len(node.children) != 2 {
			panic("Expected only two children")
		}
		compileLogf("node.children %d. [1]: %v; [0]: %v", node.ID(), node.children[1], node.children[0])
		compileLogf("node isInput %v", node.isInput())
		from := cg.df.intervals[node.children[1]].result
		to := cg.df.intervals[node.children[0]].result

		instr := letInstr{
			readFrom: from,
			writeTo:  to,
		}
		cg.instructions = append(cg.instructions, instr)

		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo.id)
	case readOp:
		// there should be only 1 child
		if len(node.children) != 1 {
			panic("Expected only one child")
		}
		compileLogf("node.children %d. [0]: %v", node.ID(), node.children[0])
		compileLogf("node isInput %v", node.isInput())
		from := cg.df.intervals[node.children[0]].result
		instr := readInstr{
			into:     op.into,
			readFrom: from,
		}
		cg.instructions = append(cg.instructions, instr)

		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo.id)
	}
}

func (cg *codegenerator) addNode(node, replacement *Node, interv *interval, i int) {
	compileLogf("Expr")
	compileLogf("Node: %x %v", node.ID(), node.op)
	compileLogf("interval %v", interv)
	writeTo := interv.result
	var reads []register
	for _, child := range node.children {
		cReplacement := cg.df.replacements[child]
		cInterv := cg.df.intervals[cReplacement]
		reads = append(reads, cInterv.result)
	}
	enterLoggingContext()
	defer leaveLoggingContext()

	var prealloc bool
	var useUnsafe bool
	// if it's not mutable, there is no chance it will be overwritten
	if node.isMutable() {
		// if the instruction calls an extern (cBLAS or cuBlas), then we should preallocate the vector
		if node.op.CallsExtern() {
			compileLogf("calls extern")
			var instr alloc
			// if i == 0 {
			// 	// if the instruction is the last instruction, we STILL want to  allocate to a new register
			// 	// if this clause is not here, the last instruction will allocate to an existing register, and overwrites any val
			// 	writeTo = register{device: writeTo.device, id: writeTo.id + 1}
			// }
			instr = newAlloc(node, writeTo)
			cg.instructions = append(cg.instructions, instr)

			cg.addInstr(node, instr)
			cg.updateLastWrites(writeTo.id)

			prealloc = true

			cg.queue = append(cg.queue, len(cg.instructions)) // no -1.
		}

		// check if any previously buffered cBLAS or cuBLAS calls need to be flushed
		// it doesn't matter if the machine isn't using a batchedBLAS. flushInstr would just be a no-op at runtime
		for _, read := range reads {
			if instrID, ok := cg.lastWrites[read.id]; ok {
				viaticum := cg.instructions[instrID] // ;) - it IS on the way
				if instr, ok := viaticum.(execOp); ok {
					if instr.op.CallsExtern() && !node.op.CallsExtern() {
						// the && bit is to make sure that if we have sequential cBLAS/cuBLAS calls,
						// we just add it to the batch.
						// sequential in this can mean several instructions apart. For example:
						//		4 	A × B 	; read %2	; write to %3
						//		 	⋮	(doesn't use %3 or %10)
						//			⋮
						//		10  Aᵀ × B	; read %3	; write to %10
						//			⋮	(doesn't use %3, or %10)
						//			⋮
						//		12 	+		; read %10	; write to %12
						//
						// It is before instruction 12 that the flush will be added. 5 and 10 are considered sequential
						if _, ok := cg.flushed[instrID]; !ok {
							cg.instructions = append(cg.instructions, flushInstr{})
							cg.addInstr(node, flushInstr{})
							cg.updateLastWrites(-1) // flush doesn't write to any register
							cg.flush()
						}
					}
				}
			}
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
		compileLogf("New Exec Op")
		instr := newExecOp(node)
		instr.readFrom = reads
		instr.writeTo = writeTo
		instr.preAllocated = prealloc
		instr.useUnsafe = useUnsafe

		cg.instructions = append(cg.instructions, instr)
		cg.addInstr(node, instr)
		cg.updateLastWrites(writeTo.id)
	}
}

func (cg *codegenerator) gen() (*program, map[*Node]register) {
	for i := len(cg.sorted) - 1; i >= 0; i-- {
		node := cg.sorted[i]
		replacement := cg.df.replacements[node]
		compileLogf("Working on %x. Replacement: %x", node.ID(), replacement.ID())

		nInterv := cg.df.intervals[replacement]
		switch {
		case node.isArg():
			cg.addArg(node, nInterv)
		case node.isStmt:
			cg.addStmt(node, nInterv)
		default:
			cg.addNode(node, replacement, nInterv, i)
		}
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
