package gorgonia

import (
	"encoding/csv"
	"fmt"
	"io"

	"github.com/pkg/errors"
)

// This file deals with the compilation from a expression graph into a program
// that is executed by an interpreter

// Compile takes a graph and outputs a program suitable for *TapeMachine to run
func Compile(g *ExprGraph) (prog *program, locMap map[*Node]register, err error) {
	compileLogf("Compiling")
	enterLoggingContext()
	defer leaveLoggingContext()

	compileLogf("sorting")
	var sortedNodes Nodes
	if sortedNodes, err = Sort(g); err != nil {
		return nil, nil, errors.Wrap(err, sortFail)
	}

	var inputs Nodes
	for _, n := range g.leaves {
		if n.isInput() {
			inputs = append(inputs, n)
		}
	}

	var outputs Nodes
	for _, root := range g.Roots() {
		outputs = append(outputs, root)
	}

	df := analyze(g, sortedNodes)

	df.intervals = buildIntervals(sortedNodes)
	ra := new(regalloc)
	ra.alloc(sortedNodes, df)

	compileLogf("Intervals: %+#v", FmtNodeMap(df.intervals))
	logCompileState(g.name, g, df)

	prog, locMap = codegen(inputs, sortedNodes, df)
	prog.locs = ra.count
	prog.df = df
	prog.g = g
	prog.sorted = sortedNodes

	return
}

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
	if sortedNodes, err = Sort(g); err != nil {
		return nil, nil, errors.Wrap(err, sortFail)
	}

	df := analyze(subgraph, sortedNodes)
	df.intervals = buildIntervals(sortedNodes)

	ra := new(regalloc)
	ra.alloc(sortedNodes, df)

	prog, locMap = codegen(inputs, sortedNodes, df)
	prog.locs = ra.count
	prog.df = df
	prog.g = g
	prog.sorted = sortedNodes

	return
}

// codegen generates the code for the VM to execute.
// It's helpful to think of len(instructions) as the instruction ID of the node
// we're currently processing
//
// Todo: This function is getting quite unwieldly. Perhaps it's time to break it down into smaller chunks?
// TODO: codegenerator struct plz kthxbai
func codegen(inputs, sorted Nodes, df *dataflow) (prog *program, locationMap map[*Node]register) {
	var instructions fragment
	g := inputs[0].g
	locationMap = make(map[*Node]register)
	instructionsMap := make(map[*Node]fragment)

	var flushQueue []int
	lastWrites := make(map[int]int)
	alreadyFlushed := make(map[int]struct{})

	updateInstrMap := func(node *Node, instr tapeInstr) {
		if instrs := instructionsMap[node]; instrs != nil {
			instrs = append(instrs, instr)
			instructionsMap[node] = instrs
		} else {
			instructionsMap[node] = fragment{instr}
		}
	}

	// every time an instruction is added to the list of instructions,
	// also add the instructionID and the register the instruction writes to.
	// This helps with determining if a flushInstruction needs to be issued.
	updateLastWrites := func(id int) {
		lastWrites[id] = len(instructions) - 1
	}

	flush := func() {
		for _, instrID := range flushQueue {
			alreadyFlushed[instrID] = struct{}{}
		}
		flushQueue = flushQueue[:0]
	}

	compileLogf("Codegen")
	enterLoggingContext()
	defer leaveLoggingContext()
	compileLogf("sorted: %d", sorted)

	for i := len(sorted) - 1; i >= 0; i-- {
		node := sorted[i]
		replacement := df.replacements[node]
		compileLogf("Working on %x. Replacement: %x", node.ID(), replacement.ID())

		nInterv := df.intervals[replacement]
		writeTo := nInterv.result
		if node.isArg() {
			// index := inputs.index(node)
			// if index >= 0 {
			compileLogf("LoadArg: %x", node.ID())
			locationMap[node] = writeTo
			instr := loadArg{
				// index:   index,
				index:   node.ID(),
				writeTo: writeTo,
			}
			instructions = append(instructions, instr)

			updateInstrMap(node, instr)
			updateLastWrites(writeTo.id)
			// }
		} else if node.isStmt {
			compileLogf("Statement")
			switch op := node.op.(type) {
			case letOp:
				// there should be only 2 chilren
				if len(node.children) != 2 {
					panic("Expected only two children")
				}
				compileLogf("node.children %d. [1]: %v; [0]: %v", node.ID(), node.children[1], node.children[0])
				compileLogf("node isInput %v", node.isInput())
				from := df.intervals[node.children[1]].result
				to := df.intervals[node.children[0]].result
				instr := letInstr{readFrom: from, writeTo: to}
				instructions = append(instructions, instr)

				updateInstrMap(node, instr)
				updateLastWrites(writeTo.id)
			case readOp:
				// there should be only 1 child
				if len(node.children) != 1 {
					panic("Expected only one child")
				}
				compileLogf("node.children %d. [0]: %v", node.ID(), node.children[0])
				compileLogf("node isInput %v", node.isInput())
				from := df.intervals[node.children[0]].result
				instr := readInstr{into: op.into, readFrom: from}
				instructions = append(instructions, instr)

				updateInstrMap(node, instr)
				updateLastWrites(writeTo.id)
			}
		} else {
			compileLogf("Expr")
			compileLogf("Node: %x", node.ID())
			var reads []register
			for _, child := range node.children {
				cReplacement := df.replacements[child]
				cInterv := df.intervals[cReplacement]
				reads = append(reads, cInterv.result)
			}
			enterLoggingContext()
			writeTo := nInterv.result

			var prealloc bool
			var useUnsafe bool
			// if it's not mutable, there is no chance it will be overwritten
			if node.isMutable() {
				// if the instruction calls an extern (cBLAS or cuBlas), then we should preallocate the vector
				if node.op.callsExtern() {
					compileLogf("calls extern")
					instr := newAlloc(node, nInterv.result)
					instructions = append(instructions, instr)

					updateInstrMap(node, instr)
					updateLastWrites(writeTo.id)

					prealloc = true

					flushQueue = append(flushQueue, len(instructions)) // no -1.
				}

				// check if any previously buffered cBLAS or cuBLAS calls need to be flushed
				// it doesn't matter if the machine isn't using a batchedBLAS. flushInstr would just be a no-op at runtime
				for _, read := range reads {
					if instrID, ok := lastWrites[read.id]; ok {
						viaticum := instructions[instrID] // ;) - it IS on the way
						if instr, ok := viaticum.(execOp); ok {
							if instr.op.callsExtern() && !node.op.callsExtern() {
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
								if _, ok := alreadyFlushed[instrID]; !ok {
									instructions = append(instructions, flushInstr{})
									updateInstrMap(node, flushInstr{})
									updateLastWrites(-1) // flush doesn't write to any register
									flush()
								}
							}
						}
					}
				}

				// check the overwrites - if the overwrite and the resulting register is the same,
				// then use unsafe options when available
				overwrites := node.op.overwriteInput()
				if overwrites >= 0 {
					compileLogf("Overwrites %d", overwrites)
					overwritten := reads[overwrites]
					compileLogf("NodeID:%d overwritten: %v, reads: %v, interval: %v", node.ID(), overwritten, nInterv.reads, nInterv.result)
					if overwritten == nInterv.result {
						compileLogf("Use unsafe")
						useUnsafe = true
					}
				}
			}

			locationMap[node] = writeTo

			// otherwise, the replacement has already been written
			if node == replacement {
				instr := newExecOp(node)
				instr.readFrom = reads
				instr.writeTo = writeTo
				instr.preAllocated = prealloc
				instr.useUnsafe = useUnsafe

				instructions = append(instructions, instr)
				updateInstrMap(node, instr)
				updateLastWrites(writeTo.id)
			}
			leaveLoggingContext()
		}
	}

	return &program{
		instructions: instructions,
		args:         len(inputs),
		g:            g,
		m:            instructionsMap,
	}, locationMap
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
			overwrites := n.op.overwriteInput()
			if overwrites >= 0 {
				row[7] = fmt.Sprintf("%d", n.children[overwrites].ID())
			}

			if n.op.callsExtern() {
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
