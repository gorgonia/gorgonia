package gorgonia

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"github.com/xtgo/set"
	"gorgonia.org/tensor"
)

type tapeMachine struct {
	ExternMetadata

	p      *program
	locMap map[*Node]register

	// "register" banks
	cpumem   []Value // Value - knows its own type and shape
	gpumem   []Value // Value of which the memories are stored in GPU memory
	cpuLocks []sync.Mutex
	gpuLocks []sync.Mutex

	// state stuff, to allow continuation after failure handling
	pc int
	execState

	// operational stuff
	bindNodesDV Nodes // nodes that require binding of DV
	watchNodes  Nodes
	watchRegs   []register
	logCh       chan logTup
	logger      *log.Logger
	buf         *bytes.Buffer
	valueFmt    string
	tabcount    int
	logFlags    byte

	sync.Mutex
	runFlags byte //  spare2: trace (copy values and put into nodes)
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
	m.cpuLocks = make([]sync.Mutex, m.p.cpulocs)
	m.gpuLocks = make([]sync.Mutex, m.p.gpulocs)

	m.init() // init ExternalMetadata

	for _, n := range m.p.g.AllNodes() {
		setEngine(n.boundTo, m.Engine)
	}
	m.execState = makeExecState(m.p)

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
	m.resetExecState()
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
	m.Lock()
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer m.Unlock()

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

func (m *tapeMachine) initRun() *sync.WaitGroup {
	wg := new(sync.WaitGroup)
	if m.logger != nil {
		m.logCh = make(chan logTup, 1024) // make it non blocking
		wg.Add(1)
		go m.startLogging(wg)
	}
	m.buf = new(bytes.Buffer)
	m.initExecState()
	return wg
}

func (m *tapeMachine) runall(errChan chan error, doneChan chan struct{}) {
	initWG := m.initRun()

	var wg sync.WaitGroup
	threads := runtime.NumCPU()
	workers := make(chan struct{}, threads)

	for t := 0; t < threads; t++ {
		workers <- struct{}{} // ensures that at any given time, there are actually THREADS available workers
		wg.Add(1)
		go m.execute(workers, errChan, &wg, t)
	}
	wg.Wait()
	// log.Println(m.buf.String())
	doneChan <- struct{}{}

	// teardowns
	if m.logCh != nil {
		m.logf("DONE")
		close(m.logCh)
		initWG.Wait()
	}
}

func (m *tapeMachine) execute(workers chan struct{}, errChan chan error, wg *sync.WaitGroup, id int) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
execloop:
	for {
		<-workers
		if m.execState.check() {
			workers <- struct{}{}
			break execloop
		}

		pnode := m.execState.next()
		if pnode == nil {
			workers <- struct{}{}
			continue
		}
		instrs := m.p.m[pnode.Node]
		for _, instr := range instrs {
			m.logf("Executing %d : %v | %v\n", pnode.index, pnode, instr)
			if err := m.executeOneInstr(instr); err != nil {
				err = errors.Wrapf(err, "pnode %d: %v\n%v", pnode.index, pnode, pnode.Value())
				errChan <- err
				m.execState.error()
				workers <- struct{}{}
				break execloop
			}
		}
		m.execState.finish(pnode)
		workers <- struct{}{}
	}
	wg.Done()
}

func (m *tapeMachine) executeOneInstr(instr tapeInstr) error {
	if err := instr.exec(m); err != nil {
		return errors.Wrapf(err, "Failed to execute instruction %v", instr)
	}

	if m.watchNaN() {
		if err := m.naughtyValues(instr, "NaN", hasNaN); err != nil {
			return err
		}
	}

	if m.watchInf() {
		if err := m.naughtyValues(instr, "Inf", hasInf); err != nil {
			return err
		}

	}
	return nil
}

func (m *tapeMachine) getValue(r register) (retVal Value) {
	switch r.device {
	case CPU:
		m.cpuLocks[r.id].Lock()
		retVal = m.cpumem[r.id]
		m.cpuLocks[r.id].Unlock()
		return retVal
	default:
		m.gpuLocks[r.id].Lock()
		retVal = m.gpumem[r.id]
		m.gpuLocks[r.id].Unlock()
		return retVal
	}
}

func (m *tapeMachine) writeValue(r register, v Value) {
	switch r.device {
	case CPU:
		m.cpuLocks[r.id].Lock()
		m.cpumem[r.id] = v
		m.cpuLocks[r.id].Unlock()
	default:
		m.gpuLocks[r.id].Lock()
		m.gpumem[r.id] = v
		m.gpuLocks[r.id].Unlock()
	}
}

func (m *tapeMachine) naughtyValues(instr tapeInstr, typ string, fn func(v Value) bool) error {
	writeTo := instr.writes().id
	id := instr.ID()

	if writeTo > 0 && id > 0 {
		v := m.getValue(instr.writes())

		if v == nil {
			return errors.Errorf(nyiFail, "converting tensor.Memory to Value", "watch"+typ)
		}
		if dev := instr.writes().device; dev != CPU {
			e := m.getEngine(dev)
			v = ScalarAsTensor(v, 2, e)
		}

		if fn(v) {
			n := m.p.g.Node(id).(*Node)
			return errors.Errorf("%v found in value. Node: %v(%x)", typ, n, n.ID())
		}
	}
	return nil
}

func (m *tapeMachine) watchedInstrLogf(instr tapeInstr, format string, attrs ...interface{}) {
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

func (m *tapeMachine) startLogging(wg *sync.WaitGroup) {
	for t := range m.logCh {
		m.logger.Printf(t.format, t.attrs...)
	}
	wg.Done()
}

func (m *tapeMachine) logf(format string, attrs ...interface{}) {
	if m.logger == nil {
		return // NO OP
	}
	m.logCh <- logTup{
		format: format,
		attrs:  attrs,
	}
}
func (m *tapeMachine) enterLogScope() {}
func (m *tapeMachine) leaveLogScope() {}

// func (m *tapeMachine) logf(format string, attrs ...interface{}) {
// 	switch {
// 	case machineDev:
// 		if m.logger != nil {
// 			goto loggercase
// 		}

// 		machineLogf(format, attrs...)
// 		break

// 	loggercase:
// 		fallthrough
// 	case m.logger != nil:
// 		s := fmt.Sprintf(format, attrs...)
// 		s = strings.Replace(s, "\n", m.buf.String(), -1)
// 		m.logger.Println(s)
// 	}
// }

// func (m *tapeMachine) enterLogScope() {
// 	if DEBUG && machineDev {
// 		enterLogScope()
// 	}
// 	m.tabcount++
// 	if m.logger != nil {
// 		reps := strings.Repeat("\t", m.tabcount)
// 		m.logger.SetPrefix(reps)
// 		m.buf.Reset()
// 		m.buf.WriteString("\n")
// 		m.buf.WriteString(reps)
// 	}
// }

// func (m *tapeMachine) leaveLogScope() {
// 	if DEBUG && machineDev {
// 		leaveLogScope()
// 	}
// 	m.tabcount--
// 	if m.tabcount < 0 {
// 		m.tabcount = 0
// 	}
// 	if m.logger != nil {
// 		reps := strings.Repeat("\t", m.tabcount)
// 		m.logger.SetPrefix(reps)
// 		m.buf.Reset()
// 		m.buf.WriteString("\n")
// 		m.buf.WriteString(reps)
// 	}
// }

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
	r            map[*Node]intervalRange
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
		m.watchedInstrLogf(instr, "%x | %T", v.Uintptr(), vt.Engine())
	} else {
		m.watchedInstrLogf(instr, "%x", v.Uintptr())
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
	return nil
}

func (instr loadArg) String() string {
	return fmt.Sprintf("loadArg %x (%v) to %v", instr.index, instr.name, instr.writeTo)
}

type execOp struct {
	op Op

	id int64 // node id

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

func (instr *execOp) exec(m *tapeMachine) error {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLogScope()
	defer m.leaveLogScope()

	// Read
	m.watchedInstrLogf(instr, "Inputs:")
	m.enterLogScope()
	inputs := make([]Value, 0, len(instr.readFrom))
	for _, reg := range instr.readFrom {
		v := m.getValue(reg)
		inputs = append(inputs, v)
		m.watchedInstrLogf(instr, m.valueFmt, v)
	}
	m.leaveLogScope()

	err := instr.execKernel(m, inputs)
	return err
}

func (instr *execOp) String() string {
	return fmt.Sprintf("%v\t%v\t%v\t%t\t%t\t%t", instr.op, instr.readFrom, instr.writeTo, instr.op.CallsExtern(), instr.useUnsafe, instr.preAllocated)
}

// flushInstr is for blastoise and cubone
type flushInstr struct{}

func (instr flushInstr) exec(m *tapeMachine) error {
	if m.WorkAvailable() == nil {
		return nil
	}
	m.ExternMetadata.Signal()
	return nil
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

func (instr deviceTransport) ID() int64         { return -1 }
func (instr deviceTransport) reads() []register { return []register{instr.from} }
func (instr deviceTransport) writes() register  { return instr.to }

func (instr deviceTransport) String() string {
	return fmt.Sprintf("memcpy(%v, %v)", instr.to, instr.from)
}

type execState struct {
	sync.RWMutex
	sorted []priorityNode
	m      map[*Node]int
	t      [][]int
	f      [][]int

	q2      chan int
	workers int32 // atomic only kthxbai
	nodes   int32 // atomic only kthxbai
	done    bool
	err     bool

	buf *bytes.Buffer
}

func makeExecState(p *program) execState {
	s := make([]priorityNode, len(p.sorted))
	m := make(map[*Node]int, len(p.sorted))
	for i := range s {
		n := p.sorted[i]
		s[i] = priorityNode{
			Node:     n,
			priority: int32(len(n.children)),
			_p:       int32(len(n.children)),
			index:    i,
		}
		m[n] = i
	}

	// "lift" tos and froms
	t := make([][]int, len(s))
	f := make([][]int, len(s))
	for k, v := range p.g.to {
		id := m[k]
		t[id] = make([]int, 0, len(v)+4) // 4 is spare
		for _, n := range v {
			nid := m[n]
			t[id] = append(t[id], nid)
		}
	}
	for _, n := range s {
		id := n.index
		f[id] = make([]int, 0, len(n.children)+4) // 4 is spare
		for _, child := range n.children {
			nid := m[child]
			f[id] = append(f[id], nid)
		}
	}

	reads := make(map[register][]int)
	writes := make(map[register][]int)

	for i := range s {
		pn := &s[i]
		n := pn.Node
		instrs := p.m[n]
		for _, instr := range instrs {
			for _, r := range instr.reads() {
				reads[r] = append(reads[r], i)
				// pn.reads.Add(r)
			}
			w := instr.writes()
			writes[w] = append(writes[w], i)
			// pn.writes.Add(w)

			if ri, ok := instr.(*readInstr); ok {
				writes[ri.readFrom] = append(writes[ri.readFrom], i)
			}
		}
	}

	// uniquify the reads and writes
	for k, v := range reads {
		reads[k] = set.Ints(v)
	}
	for k, v := range writes {
		writes[k] = set.Ints(v)
	}

	// extend the "tos" and "froms"
	for reg, wnids := range writes {
		if reg.id == -1 {
			continue
		}
		for _, nid := range wnids {
			rnids := reads[reg]
			for _, rid := range rnids {
				if rid >= nid {
					continue
				}
				f[nid] = append(f[nid], rid)
				t[rid] = append(t[rid], nid)
			}
		}
	}

	// add edges for deriv
	for i := range s {
		pn := &s[i]
		for _, deriv := range pn.derivOf {
			derivID := m[deriv]
			tos := t[derivID]
			froms := f[i]
			t[derivID] = append(tos, i)

			// SUBTLE ISSUE
			// ============
			// this is here instead of before t[deriveID] =
			// because putting this here will break cycles
			// but also creates a dependency, which will be enforced by t.
			//
			// To understand more, try moving the guard around,
			// and run TestRMSProp.
			//
			// If this guard is removed, TestRMSProp will essentially hang
			// because there is a cycle.
			// If this guard is placed before t[derivID] =...
			// then a data race will happen.
			if derivID >= i {
				continue
			}
			f[i] = append(froms, derivID)

		}
	}

	for i := range s {
		n := &s[i]
		for _, deriv := range n.derivOf {
			id := m[deriv]
			tos := t[id]
			for _, tid := range tos {
				if tid >= i {
					continue
				}
				t[tid] = append(t[tid], i)
				f[i] = append(f[i], tid)
			}
		}
	}

	for i, v := range t {
		t[i] = set.Ints(v)
	}
	for i, v := range f {
		f[i] = set.Ints(v)
	}

	for i := range s {
		s[i].priority = int32(len(f[i]))
		s[i]._p = int32(len(f[i]))
	}

	return execState{
		sorted: s,
		m:      m,
		t:      t,
		f:      f,
		nodes:  int32(len(s)),
	}
}

func (m *tapeMachine) initExecState() {
	m.execState.q2 = nil                                     // gc previous
	m.execState.q2 = make(chan int, len(m.execState.sorted)) // make new one
	m.execState.buf = m.buf

	// m.logf("%v\n", m.p)

	// loggging for failure
	// fmt.Fprintf(m.buf, "%v\n", m.p)
	// fmt.Fprintf(m.buf, "Priorities: \n")
	// for i, s := range m.execState.sorted {
	// 	fmt.Fprintf(m.buf, "\t%d: %d\n", i, s._p)
	// }
	// fmt.Fprintf(m.buf, "To:\n")
	// for i, v := range m.execState.t {
	// 	fmt.Fprintf(m.buf, "\t%d: %v\n", i, v)
	// }
	// fmt.Fprintf(m.buf, "From:\n")
	// for i, v := range m.execState.f {
	// 	fmt.Fprintf(m.buf, "\t%d: %v\n", i, v)
	// }
	// log.Printf("%v", m.buf.String())

	for _, leaf := range m.p.g.leaves {
		id := m.execState.m[leaf]
		if m.execState.sorted[id].priority > 0 {
			// there are dependencies.. not "pure" leaves
			continue
		}
		m.execState.q2 <- id
	}
	for _, c := range m.p.g.constants {
		id := m.execState.m[c]
		if m.execState.sorted[id].priority > 0 {
			// there are dependencies.. not "pure" leaves
			continue
		}
		m.execState.q2 <- id
	}
}

func (m *tapeMachine) resetExecState() {
	m.execState.q2 = nil
	m.execState.workers = 0
	m.execState.done = false
	m.execState.err = false
	m.execState.nodes = int32(len(m.execState.sorted))

	for i := range m.execState.sorted {
		m.execState.sorted[i].priority = m.execState.sorted[i]._p
		m.execState.sorted[i].status = waiting
	}
}

func (s *execState) finish(node *priorityNode) {
	to := s.t[node.index]
	for _, i := range to {
		n := &s.sorted[i]
		// this loop is necessary because to only records a single edge.
		// There may be multiple edges to the same node
		var reduction int32
		for _, j := range s.f[i] {
			if j == node.index {
				reduction--
			}
		}

		toNodePriority := atomic.AddInt32(&n.priority, reduction)
		if toNodePriority == 0 {
			// this check shouldn't be necessary
			// but I have found that the analysis algorithm
			// may lead to leaky nodes.
			status := atomic.LoadInt32(&n.status)
			if status == waiting {
				s.q2 <- n.index
			}
		}
	}

	// atomic work that things are done:
	// status is set to executed
	// workers and nodesleft decremented
	atomic.StoreInt32(&node.status, executed)
	workers := atomic.AddInt32(&s.workers, -1)
	nodes := atomic.AddInt32(&s.nodes, -1)
	if nodes == 0 && workers == 0 {
		s.Lock()
		s.done = true
		close(s.q2)
		s.Unlock()
	}
}

func (s *execState) next() (retVal *priorityNode) {
	id := <-s.q2
	if atomic.LoadInt32(&(s.sorted[id]).status) == executing {
		return nil
	}
	atomic.StoreInt32(&(s.sorted[id]).status, executing)
	atomic.AddInt32(&s.workers, 1)
	return &s.sorted[id]
}

func (s *execState) error() {
	s.Lock()
	s.err = true
	s.Unlock()
}

func (s *execState) check() bool {
	s.RLock()
	retVal := s.done || s.err
	s.RUnlock()
	return retVal
}

type priorityNode struct {
	*Node
	priority int32 // atomic only kthxbai
	status   int32 // atomic only kthxbai
	_p       int32 // original priority
	index    int
}

const (
	waiting   int32 = 0
	executed  int32 = -1
	executing int32 = 1
)

type logTup struct {
	format string
	attrs  []interface{}
}
