package gorgonia

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"strings"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

type tapeMachine struct {
	ExternMetadata

	p      *program
	locMap map[*Node]register

	// "register" banks
	cpumem []Value // Value - knows its own type and shape
	gpumem []Value // Value of which the memories are stored in GPU memory

	// state stuff, to allow continuation
	pc int

	// operational stuff
	bindNodesDV Nodes // nodes that require binding of DV
	watchNodes  Nodes
	watchRegs   []register
	logger      *log.Logger
	buf         *bytes.Buffer
	valueFmt    string
	tabcount    int
	logFlags    byte

	runFlags byte //  spare2: trace(copy values and put into nodes)
}

// NewTapeMachine creates a VM that compiles a graph into a prog.
func NewTapeMachine(g *ExprGraph, opts ...VMOpt) *tapeMachine {
	m := &tapeMachine{
		valueFmt: "%3.3g",
	}
	m.Engine = StandardEngine{}

	if b, ok := whichblas.(batchedBLAS); ok {
		m.b = b
	}

	for _, opt := range opts {
		opt(m)
	}

	m.doAlloc()

	if m.p == nil || m.locMap == nil {
		prog, locMap, err := Compile(g)
		if err != nil {
			panic(err)
		}

		m.p = prog
		m.locMap = locMap
	}
	m.cpumem = make([]Value, m.p.cpulocs)
	m.gpumem = make([]Value, m.p.gpulocs)
	m.init()
	for _, n := range m.p.g.AllNodes() {
		setEngine(n.boundTo, m.Engine)
	}

	runtime.SetFinalizer(m, finalizeTapeMachine) // a "defer" to deinitialize CUDA stuff (if using CUDA build)
	return m
}

func (m *tapeMachine) logBwd() bool { return (m.logFlags>>bwdOnly)&byte(1) == 1 }
func (m *tapeMachine) doLogBwd()    { m.logFlags |= byte(1) << bwdOnly }
func (m *tapeMachine) dontLogBwd()  { m.logFlags &= (^(byte(1) << bwdOnly)) }

func (m *tapeMachine) logFwd() bool { return (m.logFlags>>fwdOnly)&byte(1) == 1 }
func (m *tapeMachine) doLogFwd()    { m.logFlags |= byte(1) << fwdOnly }
func (m *tapeMachine) dontLogFwd()  { m.logFlags &= (^(byte(1) << fwdOnly)) }

func (m *tapeMachine) watchNaN() bool { return (m.runFlags>>watchNaN)&byte(1) == 1 }
func (m *tapeMachine) doWatchNaN()    { m.runFlags |= byte(1) << watchNaN }
func (m *tapeMachine) dontWatchNaN()  { m.runFlags &= (^(byte(1) << watchNaN)) }

func (m *tapeMachine) watchInf() bool { return (m.runFlags>>watchInf)&byte(1) == 1 }
func (m *tapeMachine) doWatchInf()    { m.runFlags |= byte(1) << watchInf }
func (m *tapeMachine) dontWatchInf()  { m.runFlags &= (^(byte(1) << watchInf)) }

func (m *tapeMachine) watchAll() bool { return (m.logFlags>>watchAll)&byte(1) == 1 }
func (m *tapeMachine) doWatchAll()    { m.logFlags |= (byte(1) << watchAll) }
func (m *tapeMachine) dontWatchAll()  { m.logFlags &= (^(byte(1) << watchAll)) }

func (m *tapeMachine) alloc() bool { return (m.runFlags>>allocVals)&byte(1) == 1 }
func (m *tapeMachine) doAlloc()    { m.runFlags |= byte(1) << allocVals }
func (m *tapeMachine) dontAlloc()  { m.runFlags &= (^(byte(1) << allocVals)) }

func (m *tapeMachine) trace() bool { return (m.runFlags>>spare2)&byte(1) == 1 }
func (m *tapeMachine) doTrace()    { m.runFlags |= byte(1) << spare2 }
func (m *tapeMachine) dontTrace()  { m.runFlags &= (^(byte(1) << spare2)) }

func (m *tapeMachine) bindDV() bool { return m.runFlags>>spare3&byte(1) == 1 }
func (m *tapeMachine) doBindDV()    { m.runFlags |= byte(1) << spare3 }
func (m *tapeMachine) dontBindDV()  { m.runFlags &= (^(byte(1) << spare3)) }

// Reset resets the run state of the machine by changing the instruction pointer back to 0
func (m *tapeMachine) Reset() {
	m.pc = 0
	m.ExternMetadata.Reset()

	for i := range m.gpumem {
		returnValue(m.gpumem[i])
		m.gpumem[i] = nil //
	}
}

func (m *tapeMachine) Close() error {
	finalizeTapeMachine(m)
	return nil
}

// Prog returns the compiled program. This would mainly be used in debugging functions
func (m *tapeMachine) Prog() *program { return m.p }

// LocMap returns the location where the Node's execution results are stored. This would mainly be used in debugging functions.
func (m *tapeMachine) LocMap() map[*Node]register { return m.locMap }

// Let wraps the Let() function of the package, with additional checks that n is in the machine
func (m *tapeMachine) Let(n *Node, be interface{}) (err error) {
	if !m.p.g.Has(n.ID()) {
		return errors.Errorf("Node %v does not exist in this graph", n)
	}

	return Let(n, be)
}

// Set wraps the Set() function of this package, with additional checks that both a and b are in the machine
func (m *tapeMachine) Set(a, b *Node) (err error) {
	if !m.p.g.Has(a.ID()) {
		return errors.Errorf("Node %v does not exist in this graph", a)
	}
	if !m.p.g.Has(b.ID()) {
		return errors.Errorf("Node %v does not exist in this graph", b)
	}

	if b.Value() != nil {
		return a.bind(b.Value())
	}

	// get the registry location
	breg := m.locMap[b]
	v := m.getValue(breg)
	if v == nil {
		return nyi("handling of tensor.Memory -> Value", "tapeMachine.Set")
	}

	machineLogf("Setting %v to %v. Read from %v Value is %v", b, a, breg, v)
	return a.bind(v)
}

// Run runs a fragment (a subset of a program).
func (m *tapeMachine) Run(frag fragment) (err error) {
	defer func() {
		if err == nil {
			m.dontAlloc()
		}
	}()

	for _, instr := range frag {
		if err = instr.exec(m); err != nil {
			return errors.Wrap(err, "Failed to carry exec()")
		}
	}
	machineLogf("Binding values based on final output")
	enterLogScope()
	for n, r := range m.locMap {
		if n.isInput() {
			continue
		}

		v := m.getValue(r)
		if v == nil {
			return nyi("converting tensor.Memory to Value", "TapeMachine.Run")
		}

		if err = n.bind(m.cpumem[r.id]); err != nil {
			return errors.Wrap(err, bindFail)
		}
	}
	leaveLogScope()
	return
}

func (m *tapeMachine) RunAll() (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer m.DoWork()

	workAvailable := m.ExternMetadata.WorkAvailable()
	syncChan := m.ExternMetadata.Sync()
	errChan := make(chan error)
	doneChan := make(chan struct{})

	go m.runall(errChan, doneChan)
	for {
		select {
		case sychronous := <-workAvailable:
			err := m.ExternMetadata.DoWork()
			if err != nil {
				return err
			}
			if sychronous {
				syncChan <- struct{}{}
			}
		case err := <-errChan:
			return errors.Wrapf(err, "PC: %d", m.pc)
		case <-doneChan:
			err := m.ExternMetadata.DoWork()
			if err != nil {
				return err
			}
			return nil
		}
	}
}

func (m *tapeMachine) runall(errChan chan error, doneChan chan struct{}) {
	for ; m.pc < len(m.p.instructions); m.pc++ {
		instr := m.p.instructions[m.pc]
		m.logf("PC %d", m.pc)
		// log.Printf("PC %d", m.pc)
		if err := instr.exec(m); err != nil {
			err = errors.Wrapf(err, "PC %d. Failed to execute instruction %v", m.pc, instr)
			errChan <- err
			return
		}
		// only proceed to check NaNs and Infs for execOp
		if _, ok := instr.(*execOp); !ok {
			continue
		}

		if m.watchNaN() {
			writeTo := instr.writes().id
			id := instr.ID()
			if writeTo > 0 && id > 0 {
				v := m.getValue(instr.writes())
				if v == nil {
					err := errors.Errorf(nyiFail, "converting tensor.Memory to Value", "watchNaN")
					errChan <- err
					return
				}

				if hasNaN(v, CPU) {
					n := m.p.g.Node(id).(*Node)
					err := errors.Errorf("NaN found in value. Node: %v(%x)", n, n.ID())
					errChan <- err
					return
				}
			}
		}

		if m.watchInf() {
			writeTo := instr.writes().id
			id := instr.ID()
			if writeTo > 0 && id > 0 {
				v := m.getValue(instr.writes())
				if v == nil {
					err := errors.Errorf(nyiFail, "converting tensor.Memory to Value", "watchInf")
					errChan <- err
					return
				}

				if hasInf(v, CPU) {
					n := m.p.g.Node(id).(*Node)
					err := errors.Errorf("Inf found in value. Node: %v(%x)", n, n.ID())
					errChan <- err
					return
				}
			}
		}
	}

	doneChan <- struct{}{}
}

func (m *tapeMachine) getValue(r register) Value {
	switch r.device {
	case CPU:
		return m.cpumem[r.id]
	default:
		return m.gpumem[r.id]
	}
}

func (m *tapeMachine) writeValue(r register, v Value) {
	switch r.device {
	case CPU:
		m.cpumem[r.id] = v
	default:
		m.gpumem[r.id] = v
	}
}

func (m *tapeMachine) watchedLogf(format string, attrs ...interface{}) {
	instr := m.p.instructions[m.pc]
	reads := instr.reads()
	writes := instr.writes()

	watched := m.watchAll()

	if !watched {
		for _, reg := range reads {
			for _, watch := range m.watchRegs {
				if reg.id == watch.id {
					watched = true
					break
				}
			}
		}
	}

	if !watched {
		for _, watch := range m.watchRegs {
			if watch.id == writes.id {
				watched = true
				break
			}
		}
	}

	// TODO: Work on watched nodes
	if !watched {

	}

	if watched {
		m.logf(format, attrs...)
	}
}

func (m *tapeMachine) logf(format string, attrs ...interface{}) {
	switch {
	case machineDev:
		if m.logger != nil {
			goto loggercase
		}

		machineLogf(format, attrs...)
		break

	loggercase:
		fallthrough
	case m.logger != nil:
		s := fmt.Sprintf(format, attrs...)
		s = strings.Replace(s, "\n", m.buf.String(), -1)
		m.logger.Println(s)
	}
}

func (m *tapeMachine) enterLogScope() {
	if DEBUG && machineDev {
		enterLogScope()
	}
	m.tabcount++
	if m.logger != nil {
		reps := strings.Repeat("\t", m.tabcount)
		m.logger.SetPrefix(reps)
		m.buf.Reset()
		m.buf.WriteString("\n")
		m.buf.WriteString(reps)
	}
}

func (m *tapeMachine) leaveLogScope() {
	if DEBUG && machineDev {
		leaveLogScope()
	}
	m.tabcount--
	if m.tabcount < 0 {
		m.tabcount = 0
	}
	if m.logger != nil {
		reps := strings.Repeat("\t", m.tabcount)
		m.logger.SetPrefix(reps)
		m.buf.Reset()
		m.buf.WriteString("\n")
		m.buf.WriteString(reps)
	}
}

/* PROGRAM */

type program struct {
	instructions fragment
	args         int
	cpulocs      int
	gpulocs      int
	cpumem       int64
	gpumem       []int64
	g            *ExprGraph         // original dag
	df           *dataflow          // dataflow analysis
	m            map[*Node]fragment // store which nodes create which instructions
	sorted       Nodes
}

func (p *program) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Instructions:\n%s\nArgs: %d | CPU Memories: %d | GPU Memories: %d\nCPU Mem: %v | GPU Mem %v\n\nNode:instructions map:\n", p.instructions, p.args, p.cpulocs, p.gpulocs, p.cpumem, p.gpumem)

	for i, n := range p.sorted {
		fmt.Fprintf(&buf, "\t%d\t%x:", i, n.ID())
		frag := p.m[n]
		for j, instr := range frag {
			if j == 0 {
				fmt.Fprintf(&buf, "\t%v\n", instr)
			} else {
				fmt.Fprintf(&buf, "\t\t%v\n", instr)
			}
		}

	}

	return buf.String()
}

// Graph enables the end user to inspect the graph (typically useful for debugging)
func (p *program) Graph() *ExprGraph { return p.g }

func (p *program) CPUMemReq() int64 { return p.cpumem }

func (p *program) GPUMemReq() []int64 {
	retVal := make([]int64, len(p.gpumem))
	copy(retVal, p.gpumem)
	return retVal
}

/* REGISTER */

type register struct {
	id     int
	device Device
}

func (r register) String() string { return fmt.Sprintf("%s%d", r.device, r.id) }

/* INSTRUCTIONS */

type tapeInstr interface {
	ID() int64 // ID is the node ID
	reads() []register
	writes() register
	exec(*tapeMachine) error
	fmt.Stringer
}

type fragment []tapeInstr

func (f fragment) String() string {
	var buf bytes.Buffer
	for i, instr := range f {
		fmt.Fprintf(&buf, "\t%d\t%s\n", i, instr)
	}
	return buf.String()
}

func (f fragment) has(want tapeInstr) bool {
	for _, instr := range f {
		if instr == want {
			return true
		}
	}
	return false
}

type alloc struct {
	id int64 // node ID
	t  hm.Type
	s  tensor.Shape

	readFrom []register
	writeTo  register
}

func newAlloc(n *Node, writeTo register) alloc {
	return alloc{
		id:      n.ID(),
		t:       n.t,
		s:       n.shape,
		writeTo: writeTo,
	}
}

func (instr alloc) ID() int64         { return instr.id }
func (instr alloc) reads() []register { return instr.readFrom }
func (instr alloc) writes() register  { return instr.writeTo }

func (instr alloc) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v", instr)
	m.enterLogScope()
	defer m.leaveLogScope()

	var dt tensor.Dtype
	if dt, err = dtypeOf(instr.t); err != nil {
		return errors.Wrapf(err, dtypeExtractionFail, instr.t)
	}

	dev := instr.writeTo.device
	var v Value
	switch dev {
	case CPU:
		v, err = makeValue(instr.t, instr.s)

	default:
		var mem tensor.Memory
		memsize := calcMemSize(dt, instr.s)
		if mem, err = m.ExternMetadata.Get(dev, memsize); err != nil {
			return errors.Wrapf(err, "Unable to allocate %v bytes from %v | %T", memsize, dev, err)
		}
		v, err = makeValueFromMem(instr.t, instr.s, mem)
	}
	if err != nil {
		return
	}
	setEngine(v, m.getEngine(dev))
	if vt, ok := v.(tensor.Tensor); ok {
		m.watchedLogf("%x | %T", v.Uintptr(), vt.Engine())
	} else {
		m.watchedLogf("%x", v.Uintptr())
	}

	m.writeValue(instr.writeTo, v)
	return nil
}

func (instr alloc) String() string {
	return fmt.Sprintf("Alloc %v%v\t\t%v", instr.t, instr.s, instr.writeTo)
}

type free struct {
	readsFrom register
}

func (instr free) ID() int64         { return -1 }
func (instr free) reads() []register { return []register{instr.readsFrom} }
func (instr free) writes() register  { return register{-1, CPU} }
func (instr free) exec(m *tapeMachine) error {
	m.logf("Executing Free %v", instr.readsFrom)
	switch instr.readsFrom.device {
	case CPU:
		return nil
	default:
		m.logf("instr.read from not CPU - %v %v %d", instr.readsFrom, instr.readsFrom.device == CPU, instr.readsFrom.device)
		mem := m.gpumem[instr.readsFrom.id]
		size := int64(mem.MemSize())

		m.Put(instr.readsFrom.device, mem, size)
		m.gpumem[instr.readsFrom.id] = nil
		return nil
	}
}
func (instr free) String() string { return fmt.Sprintf("Free %v", instr.readsFrom) }

type loadArg struct {
	index   int64
	writeTo register
	name    string
}

func (instr loadArg) ID() int64         { return instr.index }
func (instr loadArg) reads() []register { return nil }
func (instr loadArg) writes() register  { return instr.writeTo }

func (instr loadArg) exec(m *tapeMachine) error {
	m.logf("Executing %v", instr)
	m.enterLogScope()
	defer m.leaveLogScope()

	node := m.p.g.Node(instr.index).(*Node)
	m.logf("node %v", node)

	if node.boundTo == nil {
		return errors.Errorf("No value bound to node %v (%x)", node, node.ID())
	}

	var v Value
	if dv, ok := node.boundTo.(*dualValue); ok {
		v = dv.Value
	} else {
		v = node.boundTo
	}

	m.writeValue(instr.writeTo, v)
	// m.watchedLogf("Write To: %v", instr.writeTo)
	// m.watchedLogf(m.valueFmt, m.cpumem[instr.writeTo.id])
	return nil
}

func (instr loadArg) String() string {
	return fmt.Sprintf("loadArg %x (%v) to %v", instr.index, instr.name, instr.writeTo)
}

type execOp struct {
	op Op

	id int64

	readFrom []register
	writeTo  register
	size     int64 // size represents the outputsize

	preAllocated bool
	useUnsafe    bool
	useGPU       bool
}

func (instr *execOp) ID() int64         { return instr.id }
func (instr *execOp) reads() []register { return instr.readFrom }
func (instr *execOp) writes() register  { return instr.writeTo }

func newExecOp(n *Node) *execOp {
	_, useGPU := n.op.(CUDADoer)
	compileLogf("op %v uses GPU %v", n.op, useGPU)
	dt, err := dtypeOf(n.t)
	if err != nil {
		panic(err)
	}
	size := calcMemSize(dt, n.Shape())

	return &execOp{
		op:     n.op,
		id:     n.ID(),
		useGPU: useGPU,
		size:   size,
	}
}

func (instr *execOp) String() string {
	return fmt.Sprintf("%v\t%v\t%v\t%t\t%t\t%t", instr.op, instr.readFrom, instr.writeTo, instr.op.CallsExtern(), instr.useUnsafe, instr.preAllocated)
}

// flushInstr is for blastoise and cubone
type flushInstr struct{}

func (instr flushInstr) exec(m *tapeMachine) error {
	m.logf("Executing DoWork")
	return m.ExternMetadata.DoWork()
}

func (instr flushInstr) ID() int64         { return -1 }
func (instr flushInstr) reads() []register { return nil }
func (instr flushInstr) writes() register  { return register{-1, CPU} }
func (instr flushInstr) String() string    { return "DoWork" }

type letInstr struct {
	readFrom register
	writeTo  register
}

func (instr letInstr) ID() int64               { return -1 }
func (instr letInstr) reads() []register       { return []register{instr.readFrom} }
func (instr letInstr) writes() register        { return instr.writeTo }
func (instr letInstr) exec(*tapeMachine) error { return nil }

func (instr letInstr) String() string {
	return fmt.Sprintf("LET %v = %v", instr.writeTo, instr.readFrom)
}

type readInstr struct {
	readFrom register
	into     *Value

	// required to convert tensor.Memory to Value
	t hm.Type
	s tensor.Shape
}

func (instr *readInstr) ID() int64         { return -1 }
func (instr *readInstr) reads() []register { return []register{instr.readFrom} }
func (instr *readInstr) writes() register  { return register{-1, CPU} }
func (instr *readInstr) exec(m *tapeMachine) (err error) {
	m.logf("Executing READ - read from %v into %v", instr.readFrom, instr.into)
	v := m.getValue(instr.readFrom)
	if v == nil {
		return nyi("value of nil", "readInstr.exec")
	}

	v2, err := CloneValue(v)
	if err != nil {
		return errors.Wrap(err, cloneFail)
	}

	*instr.into = v2
	return nil
}

func (instr *readInstr) String() string {
	return fmt.Sprintf("Read %v into %p", instr.readFrom, instr.into)
}

type deviceTransport struct {
	from, to register
}

func (instr deviceTransport) ID() int64 { return -1 }
func (instr deviceTransport) reads() []register {
	return []register{instr.from}
}
func (instr deviceTransport) writes() register { return instr.to }

func (instr deviceTransport) String() string {
	return fmt.Sprintf("memcpy(%v, %v)", instr.to, instr.from)
}
