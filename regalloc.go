package gorgonia

import (
	"fmt"

	"github.com/xtgo/set"
)

// this file holds all the code that relates to register allocation
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
	// compileLogf("%v live at %d", i, id)
	if i.start <= id && id < i.end {
		return true
	}
	return false
}

func (i *interval) lastUse() int {
	if len(i.usePositions) == 0 {
		return -1
	}

	// if !sort.IntsAreSorted(i.usePositions) {
	// 	sort.Ints(i.usePositions)
	// }
	return i.usePositions[len(i.usePositions)-1]
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

type regalloc struct {
	cpucount      int
	gpucount      int
	instructionID int
	df            *dataflow
}

func newRegalloc(df *dataflow) *regalloc {
	return &regalloc{
		df: df,
	}
}

func (ra *regalloc) newReg(device Device) register {
	var out register
	switch device {
	case CPU:
		out = register{ra.cpucount, device}
		ra.cpucount++
	default:
		out = register{ra.gpucount, device}
		ra.gpucount++

	}
	return out
}

func (ra *regalloc) allocArg(nInterv *interval) {
	nInterv.result = ra.newReg(CPU)
}

func (ra *regalloc) allocMutableOp(node *Node, nInterv *interval) {
	// create new write to if overwriteInput and the used register is stil live
	compileLogf("Allocating MutableOp NodeID: %x returns pointer", node.ID())
	compileLogf("Op: %v", node.op)
	enterLogScope()
	defer leaveLogScope()

	var writeTo register
	var reads []*interval

	var children Nodes
	var ok bool
	if children, ok = ra.df.devTransChildren[node]; !ok {
		compileLogf("replacement children not found")
		children = node.children
	}
	for _, child := range children {
		cReplace := ra.df.replacements[child]
		repInterv := ra.df.intervals[cReplace]
		reads = append(reads, repInterv)
	}
	compileLogf("Read %v", reads)

	var letStmts Nodes
	it := node.g.To(node.ID())
	for it.Next() {
		parent := it.Node()

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
	var onDev bool
	switch node.op.(type) {
	case CUDADoer:
		onDev = true
	case CLDoer:
		onDev = true
	default:
	}

	if overwrites >= 0 {
		overwriteReg := reads[overwrites].result
		overwriteDev := overwriteReg.device
		overwrittenIsLive := reads[overwrites].liveAt(ra.instructionID)
		compileLogf("Overwrites : %v ", overwrites)
		compileLogf("Overwritten (%v) is live at %d? %t", reads[overwrites], ra.instructionID, overwrittenIsLive)
		compileLogf("Let Statements: %d | %v", len(letStmts), reads[overwrites])

		// If the overwritten is not live, and the node does not call external processes (obiviating the need to prealloc)
		// then we can directly overwrite the register.
		if len(letStmts) == 1 || !overwrittenIsLive {

			switch {
			case onDev && overwriteDev != CPU:
				// if overwritten reg is on external device and op will execute on external device
				// then safe to overwrite
				writeTo = overwriteReg
			case !node.op.CallsExtern() && overwriteDev == CPU:
				// original case:
				// if the op doesn't call an extern, and is executed on CPU
				// safe to overwrite
				writeTo = overwriteReg
			case onDev:
				// new register otherwise
				writeTo = ra.newReg(Device(0))
			case !onDev:
				// new register otherwise
				writeTo = ra.newReg(CPU)
			}

		} else {
			if onDev {
				writeTo = ra.newReg(Device(0))
			} else {
				writeTo = ra.newReg(CPU)
			}
		}
	} else {
		compileLogf("New register")
		if onDev {
			writeTo = ra.newReg(Device(0))
		} else {
			writeTo = ra.newReg(CPU)
		}
	}

	for _, r := range reads {
		nInterv.reads = append(nInterv.reads, r.result)
	}
	nInterv.result = writeTo
	compileLogf("%v: %v", node.op, nInterv)
}

func (ra *regalloc) allocImmutableOp(node *Node, nInterv *interval) {
	compileLogf("Allocating Immutable Op")
	enterLogScope()
	defer leaveLogScope()

	var writeTo register
	var reads []*interval

	var children Nodes
	var ok bool
	if children, ok = ra.df.devTransChildren[node]; !ok {
		children = node.children
	}
	for _, child := range children {
		cReplace := ra.df.replacements[child]
		repInterv := ra.df.intervals[cReplace]
		reads = append(reads, repInterv)
	}

	compileLogf("NodeID: %x does not returns pointer", node.ID())
	if _, ok := node.op.(CUDADoer); ok {
		writeTo = ra.newReg(Device(0))
	} else {
		writeTo = ra.newReg(CPU)
	}

	for _, r := range reads {
		nInterv.reads = append(nInterv.reads, r.result)
	}
	nInterv.result = writeTo
}

func (ra *regalloc) allocStatement(node *Node, nInterv *interval) {
	var writeTo register
	switch op := node.op.(type) {
	case devTrans:
		writeTo = ra.newReg(op.to)
	}
	nInterv.result = writeTo
}

func (ra *regalloc) alloc(sorted Nodes) {
	compileLogf("Allocating registers")
	enterLogScope()
	defer leaveLogScope()

	for i, node := range sorted {
		ra.instructionID = i

		replacement := ra.df.replacements[node]
		nInterv := ra.df.intervals[replacement]

		compileLogf("replacement %v, interval %v", replacement, nInterv)

		if node != replacement {
			compileLogf("Merging")
			ra.df.intervals[node].merge(nInterv)
		}
		compileLogf("Working on %v(%x). InstructionID: %d", node, node.ID(), ra.instructionID)

		switch {
		case node.isArg():
			ra.allocArg(nInterv)
		case node.isStmt:
			ra.allocStatement(node, nInterv)
		case node.op.ReturnsPtr():
			ra.allocMutableOp(node, nInterv)
		default:
			ra.allocImmutableOp(node, nInterv)
		}
		compileLogf("n: %x; result: %v; reads: %v", node.ID(), nInterv.result, nInterv.reads)
		// ra.instructionID++
	}
}
