package gorgonia

import (
	"fmt"
	"io/ioutil"

	"github.com/xtgo/set"
)

// this file holds all the code that relates to registrer allocation
// a lot of the code is shamelessly copied from my previous HIL work, the thirteenthfloor
// TODO: cleanup

type interval struct {
	start, end int

	result       register
	reads        []register
	ranges       []intervalRange
	usePositions []int
}

func newInterval() *interval {
	retVal := &interval{
		start: -1,
		end:   -1,
	}

	// switch vr.(type) {
	// case *ssa.StaticSingleAssignment:
	// 	retVal.isDestination = true
	// case *ssa.Phi:
	// case *ssa.Pop:
	// 	retVal.isDestination = true
	// default:
	// }
	return retVal
}

func (i *interval) String() string {
	return fmt.Sprintf("%s | %d - %d | %v", i.result, i.start, i.end, i.usePositions)
}

func (i *interval) setFrom(from int) {
	if i.start == -1 || (from < i.start && from >= 0) {
		i.start = from
	}
}

func (i *interval) fix() {
	if len(i.usePositions) == 0 {
		return
	}
	i.usePositions = set.Ints(i.usePositions)
	i.end = i.usePositions[len(i.usePositions)-1]

	for _, r := range i.ranges {
		if r.to > i.end {
			i.end = r.to
		}
	}
}

// func (i *interval) setTo(to int) {
// 	if to < i.start {
// 		// invalid To
// 		panic("to  < start")
// 	}

// 	if to < i.end {
// 		uPs := i.usePositions
// 		sort.Ints(i.usePositions)

// 		maxUP := uPs[len(uPs)-1]
// 		if to >= maxUP {
// 			i.end = to
// 		}
// 	} else if to > i.end {
// 		i.end = to
// 	}
// }

func (i *interval) addRange(from, to int) {
	if to < from {
		panic("to < from") // note: to == from is a valid interval range
	}

	r := intervalRange{from, to}

	// because I'm lazy to create a intervalRangeSet type, we'll just iterate and check
	for _, ra := range i.ranges {
		if r == ra {
			return
		}
	}

	i.ranges = append(i.ranges, r)

	// set the end property
	if to > i.end {
		i.end = to
	}

	i.setFrom(from)
}

// added so only unique usePositions are added
func (i *interval) addUsePositions(up int) {
	// for _, u := range i.usePositions {
	// 	if u == up {
	// 		return
	// 	}
	// }

	// i.usePositions = append(i.usePositions, up)
	// if i.end < up {
	// 	i.setTo(up)
	// }

	i.usePositions = append(i.usePositions, up)
}

func (i *interval) noUsePositions() bool {
	if len(i.usePositions) == 0 || i.usePositions == nil {
		return true
	}
	return false
}

// inclusive of start, but exclusive of end
func (i *interval) liveAt(id int) bool {
	compileLogf("%v live at %d", i, id)
	if i.start <= id && id < i.end {
		return true
	}
	return false
}

func (i *interval) merge(other *interval) {
	if other.start < i.start && other.start >= 0 {
		i.start = other.start
	}

	if other.end > i.end {
		i.end = other.end
	}

	for _, r := range other.ranges {
		i.addRange(r.from, r.to)
	}

	i.usePositions = append(i.usePositions, other.usePositions...)
	i.usePositions = set.Ints(i.usePositions)

}

type intervalRange struct {
	from, to int
}

/*
	Notes on handling the live set:

	1. We load all the SSAs listed in the block's LiveIn
	2. Then we load all the SSAs used as input in this block Phi nodes
		- The reason for this is so that those SSAs can have intervals created
		  that are live in this block (well, they are kinda live)
	3. These input SSAs are temporary only, because a path-dependent liveset will be calculated below

	Consider a CFG that looks like this:

                           BLOCK 1           BLOCK 3
                           +-------+        +-------+
                     +---->| x = 1 +------->| y = 3 +----------------+
        BLOCK 0      |     +-------+        | use x |                v  BLOCK 4
       +-------+     |                      +-------+              +-------------+
       |       |+----+                                             | x = ϕ(1, 2) |
       +-------+     |     BLOCK 2                                 +-------------+
                     |     +-------+                                 ^
                     +---->| x = 2 +---------------------------------+
                           +-------+

	`x = 1` needs to be live in BLOCK 1, BLOCK 3 and BLOCK 4
	`x = 2` needs to be live in BLOCK 2 and BLOCK 4.

	The solution: in BLOCK 4, load `x = 1` and `x = 2` so they can be considered live in Block 4.

	The interval building process comes to BLOCK 3 next. It considers the SSAs that are live in BLOCK 4.
	If `x = 2` is live in BLOCK 4, it's Bad News with capital letters (see comment below).

	The solution: remove the InputSSAs of the Phi nodes when we're leaving this block.
*/
// TODO: rephrase above to fit this package's function.
// It's like the above, but without basic blocks, phi nodes, etc, making it a LOT simpler
func buildIntervals(sorted Nodes) map[*Node]*interval {
	intervals := make(map[*Node]*interval)

	var g *ExprGraph
	for _, n := range sorted {
		if g == nil && n.g != nil {
			g = n.g
		}

		intervals[n] = newInterval()
	}
	instructions := len(sorted)
	for i, n := range sorted {
		instrNum := instructions - 1 - i
		nInter := intervals[n]

		// inputs will be live the entire program
		if n.isInput() {
			nInter.addRange(instrNum, instructions)
			continue
		}
		nInter.addRange(instrNum, instrNum)
		// nInter.setFrom(instrNum)
		// nInter.setTo(instrNum)

		for _, child := range n.children {
			iv, ok := intervals[child]
			if !ok {
				parents := g.to[n]
				for i, from := range parents {
					ioutil.WriteFile(fmt.Sprintf("n_%d.dot", i), []byte(from.ToDot()), 0644)
				}
			}

			iv.addUsePositions(instrNum)
			// iv.setTo(instrNum)
		}
	}

	for _, iv := range intervals {
		iv.fix()
	}

	return intervals
}

type regalloc struct {
	count int
}

func (ra *regalloc) newReg(device Device) register {
	out := register{ra.count, device}
	ra.count++
	return out
}

func (ra *regalloc) alloc(sorted Nodes, df *dataflow) {
	compileLogf("Allocating registers")
	enterLoggingContext()
	defer leaveLoggingContext()

	var instructionID int
	for i := len(sorted) - 1; i >= 0; i-- {
		node := sorted[i]

		replacement := df.replacements[node]
		nInterv := df.intervals[replacement]

		if node != replacement {
			compileLogf("Merging")
			// nInterv.merge(df.intervals[node])
			df.intervals[node].merge(nInterv)
		}
		compileLogf("Working on %x. InstructionID: %d", node.ID(), instructionID)
		enterLoggingContext()
		if node.isArg() {
			nInterv.result = ra.newReg(CPU)
		} else {
			compileLogf("not arg...")
			var reads []*interval
			for _, child := range node.children {
				cReplace := df.replacements[child]
				repInterv := df.intervals[cReplace]
				reads = append(reads, repInterv)
			}

			var writeTo register
			if node.op.ReturnsPtr() {
				// create new write to if overwriteInput and the used register is stil live
				compileLogf("NodeID: %x returns pointer", node.ID())
				compileLogf("Op: %v", node.op)
				enterLoggingContext()

				var letStmts Nodes
				for _, parent := range node.g.To(node) {
					n := parent.(*Node)
					compileLogf("Parent: %v | %T", n, n.op)
					if n.isStmt {
						// compileLogf("isStmt")
						if _, ok := n.op.(letOp); ok {
							letStmts = append(letStmts, n)
						}
					}
				}

				overwrites := node.op.OverwritesInput()
				if overwrites >= 0 {
					overwrittenIsLive := reads[overwrites].liveAt(instructionID)
					compileLogf("Overwrites : %v ", overwrites)
					compileLogf("Overwritten (%v) is live at %d? %t", reads[overwrites], instructionID, overwrittenIsLive)

					compileLogf("Let Statements: %d | %v", len(letStmts), reads[overwrites])
					if len(letStmts) == 1 || !overwrittenIsLive {
						writeTo = reads[overwrites].result
					} else {
						if _, ok := node.op.(CUDADoer); ok {
							writeTo = ra.newReg(Device(0))
						} else {
							writeTo = ra.newReg(CPU)
						}
					}
				} else {
					compileLogf("New register")
					if _, ok := node.op.(CUDADoer); ok {
						writeTo = ra.newReg(Device(0))
					} else {
						writeTo = ra.newReg(CPU)
					}
				}
				leaveLoggingContext()
			} else {
				compileLogf("NodeID: %x does not returns pointer", node.ID())
				if _, ok := node.op.(CUDADoer); ok {
					writeTo = ra.newReg(Device(0))
				} else {
					writeTo = ra.newReg(CPU)
				}
			}

			for _, r := range reads {
				nInterv.reads = append(nInterv.reads, r.result)
			}
			nInterv.result = writeTo
		}
		leaveLoggingContext()
		compileLogf("n: %x; result: %v; reads: %v", node.ID(), nInterv.result, nInterv.reads)
		instructionID++
	}
}
