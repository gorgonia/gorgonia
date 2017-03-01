package gorgonia

import (
	"bytes"
	"fmt"
	"log"
	"runtime"
	"strings"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type tapeMachine struct {
	*ExternMetadata

	p      *program
	cpumem []Value
	gpumem []Memory
	locMap map[*Node]register

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

// NewTapeMachine creates a VM that executes a pre-compiled graph.
func NewTapeMachine(prog *program, locMap map[*Node]register, opts ...VMOpt) *tapeMachine {
	m := &tapeMachine{
		ExternMetadata: new(ExternMetadata),
		p:              prog,
		locMap:         locMap,
		cpumem:         make([]Value, prog.cpulocs),
		gpumem:         make([]Memory, prog.gpulocs),
		valueFmt:       "%3.3f",
	}

	if b, ok := whichblas.(batchedBLAS); ok {
		m.b = b
	}

	for _, opt := range opts {
		opt(m)
	}

	m.doAlloc()

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
func (m *tapeMachine) Reset() { m.pc = 0 }

// Let wraps the Let() function of the package, with additional checks that n is in the machine
func (m *tapeMachine) Let(n *Node, be interface{}) (err error) {
	if !m.p.g.Has(n) {
		return errors.Errorf("Node %v does not exist in this graph", n)
	}

	return Let(n, be)
}

// Set wraps the Set() function of this package, with additional checks that both a and b are in the machine
func (m *tapeMachine) Set(a, b *Node) (err error) {
	if !m.p.g.Has(a) {
		return errors.Errorf("Node %v does not exist in this graph", a)
	}
	if !m.p.g.Has(b) {
		return errors.Errorf("Node %v does not exist in this graph", b)
	}

	// get the registry location
	// areg := m.locMap[a]
	breg := m.locMap[b]
	v := m.cpumem[breg.id]
	machineLogf("Setting %v to %v. Read from %v Value is %v", b, a, breg, v)

	return a.bind(v)
}

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
	enterLoggingContext()
	for n, r := range m.locMap {
		if n.isInput() {
			continue
		}

		if err = n.bind(m.cpumem[r.id]); err != nil {
			return errors.Wrap(err, bindFail)
		}
	}
	leaveLoggingContext()
	return
}

func (m *tapeMachine) RunAll() (err error) {
	defer func() {
		if err == nil {
			m.dontAlloc()
		}
	}()

	// defer func() {
	// 	if r := recover(); r != nil {
	// 		log.Printf("m.ExternMetadata %v", m.ExternMetadata)
	// 		panic(r)
	// 	}
	// }()
	workAvailable := m.ExternMetadata.WorkAvailable()

	for {
		select {
		case <-workAvailable:
			m.ExternMetadata.DoWork()
		default:
			if m.pc >= len(m.p.instructions) {
				return
			}
			if err = m.runone(); err != nil {
				return errors.Wrapf(err, "PC: %d", m.pc)
			}
			m.pc++
		}
	}
	m.DoAllWork()

	return
}

func (m *tapeMachine) runone() (err error) {
	instr := m.p.instructions[m.pc]
	if err = instr.exec(m); err != nil {
		return errors.Wrap(err, "Failed to carry exec()")
	}

	if m.watchNaN() {
		writeTo := instr.writes().id
		id := instr.ID()
		if writeTo > 0 && id > 0 {
			v := m.cpumem[writeTo]
			n := m.p.g.Node(id).(*Node)

			if hasNaN(v) {
				return errors.Errorf("NaN found in value. Node: %v(%x)", n, n.ID())
			}
		}
	}

	if m.watchInf() {
		writeTo := instr.writes().id
		id := instr.ID()
		if writeTo > 0 && id > 0 {
			v := m.cpumem[writeTo]
			n := m.p.g.Node(id).(*Node)
			if hasInf(v) {
				return errors.Errorf("Inf found in value. Node: %v(%x)", n, n.ID())
			}
		}
	}
	return nil
}

func (m *tapeMachine) getValue(r register) (Value, Memory) {
	switch r.device {
	case CPU:
		return m.cpumem[r.id], nil
	default:
		mem := m.gpumem[r.id]
		if v, ok := mem.(Value); ok {
			return v, mem
		}
		return nil, mem // needs copying
	}
}

func (m *tapeMachine) getMemory(r register) Memory {
	switch r.device {
	case CPU:
		return m.cpumem[r.id].(Memory)
	default:
		return m.gpumem[r.id]
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

func (m *tapeMachine) enterLoggingContext() {
	if DEBUG && machineDev {
		enterLoggingContext()
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

func (m *tapeMachine) leaveLoggingContext() {
	if DEBUG && machineDev {
		leaveLoggingContext()
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
	g            *ExprGraph         // original dag
	df           *dataflow          // dataflow analysis
	m            map[*Node]fragment // store which nodes create which instructions
	sorted       Nodes
}

func (p *program) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Instructions:\n%s\nArgs: %d | Memories: %d\n\nNode:instructions map:\n", p.instructions, p.args, p.cpulocs)

	for k, v := range p.m {
		fmt.Fprintf(&buf, "\t%x:", k.ID())
		for i, instr := range v {
			if i == 0 {
				fmt.Fprintf(&buf, "\t%v\n", instr)
			} else {
				fmt.Fprintf(&buf, "\t\t%v\n", instr)
			}
		}
	}
	return buf.String()
}

/* REGISTER */

type register struct {
	id     int
	device Device
}

func (r register) String() string { return fmt.Sprintf("%s%d", r.device, r.id) }

/* INSTRUCTIONS */

type tapeInstr interface {
	ID() int // ID is the node ID
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
	id int // node ID
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

func (instr alloc) ID() int           { return instr.id }
func (instr alloc) reads() []register { return instr.readFrom }
func (instr alloc) writes() register  { return instr.writeTo }

func (instr alloc) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v", instr)

	dest := instr.writeTo.id
	device := instr.writeTo.device

	// check
	switch device {
	case CPU:
		var have, want hm.Type
		if m.cpumem[dest] == nil {
			goto mustalloc
		}
		have = TypeOf(m.cpumem[dest])
		want = instr.t

		if !m.alloc() && have == want {
			m.logf("Already prealloc")

			return
		}
	mustalloc:
		// check first if there is already a value bound to the node.
		node := m.p.g.Node(instr.id).(*Node)
		if node.boundTo != nil {
			switch v := node.boundTo.(type) {
			case tensor.Tensor:
				m.cpumem[dest] = v
				return nil
			case *dualValue:
				if tv, ok := v.Value.(tensor.Tensor); ok {
					m.cpumem[dest] = tv
					return nil
				}
			case Scalar:
				// do nothing
			}
		}

		machineLogf("Have to allocate %v in register %v", instr.t, instr.writeTo)
		var tt TensorType
		var ok bool
		if tt, ok = instr.t.(TensorType); !ok {
			return errors.New("Alloc only allocates tensor types")

			// allocate a "scalar" vector
		}

		var dt tensor.Dtype
		if dt, ok = tt.Of.(tensor.Dtype); !ok {
			return errors.Errorf("No dtype to allocate. Type: %T", tt.Of)
		}

		//TODO: runtime shape check
		t := tensor.New(tensor.Of(dt), tensor.WithShape(instr.s...))
		m.cpumem[dest] = t
		return
	default:
		if m.gpumem[dest] != nil {
			// check mem info
			// return if already as expected
		}
		// check if there is already a value bound to node
		node := m.p.g.Node(instr.id).(*Node)
		if node.boundTo != nil {
			switch v := node.boundTo.(type) {
			case tensor.Tensor:
			case *dualValue:
				if tv, ok := v.Value.(tensor.Tensor); ok {
					machineLogf("TV %v", tv)
				}
			}
		}
		machineLogf("Have to allocate %v in register %v")
		var tt TensorType
		var ok bool
		if tt, ok = instr.t.(TensorType); !ok {
			return errors.New("Alloc only allocates tensor types")

			// allocate a "scalar" vector
		}

		var dt tensor.Dtype
		if dt, ok = tt.Of.(tensor.Dtype); !ok {
			return errors.Errorf("No dtype to allocate. Type: %T", tt.Of)
		}

		size := int(dt.Size()) * instr.s.TotalSize()
		var mem Memory
		if mem, err = device.Alloc(m, int64(size)); err != nil {
			return errors.Wrapf(err, "Failed to allocate %d bytes on %v", size, device)
		}

		m.gpumem[dest] = mem
		return nil
	}
}

func (instr alloc) String() string {
	return fmt.Sprintf("Alloc %v\t\t%v", instr.t, instr.writeTo)
}

type free struct {
	readsFrom register
}

func (instr free) ID() int           { return -1 }
func (instr free) reads() []register { return []register{instr.readsFrom} }
func (instr free) writes() register  { return register{-1, CPU} }
func (instr free) exec(m *tapeMachine) error {
	switch instr.readsFrom.device {
	case CPU:
		return nil
	default:
		mem := m.gpumem[instr.readsFrom.id]
		if err := instr.readsFrom.device.Free(m, mem); err != nil {
			return err
		}
		m.gpumem[instr.readsFrom.id] = nil
		return nil
	}
}
func (instr free) String() string { return fmt.Sprintf("Free %v", instr.readsFrom) }

type loadArg struct {
	index   int
	writeTo register
}

func (instr loadArg) ID() int           { return instr.index }
func (instr loadArg) reads() []register { return nil }
func (instr loadArg) writes() register  { return instr.writeTo }

func (instr loadArg) exec(m *tapeMachine) error {
	m.logf("Executing %v", instr)
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	node := m.p.g.Node(instr.index).(*Node)

	if node.boundTo == nil {
		return errors.Errorf("No value bound to node %v (%x)", node, node.ID())
	}

	var v Value
	if dv, ok := node.boundTo.(*dualValue); ok {
		v = dv.Value
	} else {
		v = node.boundTo
	}

	m.cpumem[instr.writeTo.id] = v
	m.watchedLogf("Write To: %v", instr.writeTo)
	m.watchedLogf(m.valueFmt, m.cpumem[instr.writeTo.id])
	return nil
}

func (instr loadArg) String() string {
	return fmt.Sprintf("loadArg %x to %v", instr.index, instr.writeTo)
}

type execOp struct {
	op          Op
	inputTypes  hm.Types
	outputType  hm.Type
	outputShape tensor.Shape

	id int

	readFrom []register
	writeTo  register

	preAllocated bool
	useUnsafe    bool
	useGPU       bool
}

func (instr execOp) ID() int           { return instr.id }
func (instr execOp) reads() []register { return instr.readFrom }
func (instr execOp) writes() register  { return instr.writeTo }

func newExecOp(n *Node) *execOp {
	var inputTypes hm.Types
	for _, child := range n.children {
		inputTypes = append(inputTypes, child.t)
	}

	_, useGPU := n.op.(CUDADoer)

	return &execOp{
		op:          n.op,
		id:          n.ID(),
		inputTypes:  inputTypes,
		outputType:  n.t,
		outputShape: n.shape,
		useGPU:      useGPU,
	}
}

func (instr execOp) String() string {
	return fmt.Sprintf("%v\t%v\t%v\t%v\t%t\t%t\t%t", instr.op, instr.readFrom, instr.writeTo, instr.inputTypes, instr.op.CallsExtern(), instr.useUnsafe, instr.preAllocated)
}

// flushInstr is for blastoise and cubone
type flushInstr struct{}

func (instr flushInstr) exec(m *tapeMachine) error {
	if m.b == nil {
		return nil
	}
	m.ExternMetadata.DoWork()
	return nil
}

func (instr flushInstr) ID() int           { return -1 }
func (instr flushInstr) reads() []register { return nil }
func (instr flushInstr) writes() register  { return register{-1, CPU} }
func (instr flushInstr) String() string    { return "DoWork" }

type letInstr struct {
	readFrom register
	writeTo  register
}

func (instr letInstr) ID() int                 { return -1 }
func (instr letInstr) reads() []register       { return []register{instr.readFrom} }
func (instr letInstr) writes() register        { return instr.writeTo }
func (instr letInstr) exec(*tapeMachine) error { return nil }

func (instr letInstr) String() string {
	return fmt.Sprintf("LET %v = %v", instr.writeTo, instr.readFrom)
}

type readInstr struct {
	readFrom register
	into     *Value
}

func (instr readInstr) ID() int           { return -1 }
func (instr readInstr) reads() []register { return []register{instr.readFrom} }
func (instr readInstr) writes() register  { return register{-1, CPU} }
func (instr readInstr) exec(m *tapeMachine) error {
	v := m.cpumem[instr.readFrom.id]
	v2, err := CloneValue(v)
	if err != nil {
		return errors.Wrap(err, cloneFail)
	}

	*instr.into = v2
	return nil
}

func (instr readInstr) String() string {
	return fmt.Sprintf("Read %v into %p", instr.readFrom, instr.into)
}

type deviceTransport struct {
	from, to   Device
	registerID int
}

func (instr deviceTransport) ID() int { return -1 }
func (instr deviceTransport) reads() []register {
	return []register{register{id: instr.registerID, device: instr.from}}
}
func (instr deviceTransport) writes() register { return register{instr.registerID, instr.to} }
func (instr deviceTransport) exec(m *tapeMachine) error {

	return nil
}
