package gorgonia

import (
	"bytes"
	"fmt"
	"log"
	"strings"

	"github.com/chewxy/gorgonia/tensor/types"
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

type tapeMachine struct {
	p       *program
	storage []Value
	locMap  map[*Node]register
	b       batchedBLAS

	// state stuff, to allow continuation
	pc int

	// logging stuff
	watchNodes Nodes
	watchRegs  []register
	logger     *log.Logger
	buf        *bytes.Buffer
	valueFmt   string
	tabcount   int
	logFlags   byte

	runFlags byte //  spare2: trace(copy values and put into nodes)
}

func NewTapeMachine(prog *program, locMap map[*Node]register, opts ...VMOpt) *tapeMachine {
	m := &tapeMachine{
		p:        prog,
		locMap:   locMap,
		storage:  make([]Value, prog.locs),
		valueFmt: "%3.3f",
	}

	if b, ok := whichblas.(batchedBLAS); ok {
		m.b = b
	}

	for _, opt := range opts {
		opt(m)
	}

	m.doAlloc()

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

// Let wraps the Let() function of the package, with additional checks that n is in the machine
func (m *tapeMachine) Let(n *Node, be interface{}) (err error) {
	if !m.p.g.Has(n) {
		return errors.Errorf("Node %v does not exist in this graph", n)
	}

	return Let(n, be)
}

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
	v := m.storage[breg.id]
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

		if err = n.bind(m.storage[r.id]); err != nil {
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

	for ; m.pc < len(m.p.instructions); m.pc++ {
		instr := m.p.instructions[m.pc]
		if err = instr.exec(m); err != nil {
			return errors.Wrap(err, "Failed to carry exec()")
		}

		if m.watchNaN() {
			writeTo := instr.writes().id
			id := instr.ID()
			if writeTo > 0 && id > 0 {
				v := m.storage[writeTo]
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
				v := m.storage[writeTo]
				n := m.p.g.Node(id).(*Node)
				if hasInf(v) {
					log.Printf("Reads: %v", instr.reads())
					return errors.Errorf("Inf found in value. Node: %v(%x)", n, n.ID())
				}
			}
		}
	}

	// re-bind the values to the nodes
	// machineLogf("Binding values based on final output")
	// enterLoggingContext()
	// for n, r := range m.locMap {
	// 	if n.isInput() {
	// 		continue
	// 	}

	// 	if err = n.bind(m.storage[r.id]); err != nil {
	// 		return
	// 	}
	// }
	// leaveLoggingContext()
	return
}

func (m *tapeMachine) Reset() {
	m.pc = 0
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
		machineLogf(format, attrs...)

		if m.logger != nil {
			goto loggercase
		}
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
	if DEBUG {
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
	if DEBUG {
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
	locs         int
	g            *ExprGraph         // original dag
	df           *dataflow          // dataflow analysis
	m            map[*Node]fragment // store which nodes create which instructions
	sorted       Nodes
}

func (p *program) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "Instructions:\n%s\nArgs: %d | Memories: %d\n\nNode:instructions map:\n", p.instructions, p.args, p.locs)

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
	ID() int
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

type alloc struct {
	id int // node ID
	t  hm.Type
	s  types.Shape

	readFrom []register
	writeTo  register
}

func newAlloc(n *Node, writeTo register) alloc {
	return alloc{
		id:      n.ID(),
		t:       hm.Prune(n.t),
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

	// check
	var have, want hm.Type
	if m.storage[dest] == nil {
		goto mustalloc
	}
	have = m.storage[dest].Type()
	want = instr.t

	if !m.alloc() && have.Eq(want) {
		machineLogf("Already preallocated!")
		m.logf("Already prealloc")

		return
	}

mustalloc:
	// check first if there is already a value bound to the node.
	node := m.p.g.Node(instr.id).(*Node)
	if node.boundTo != nil {
		switch v := node.boundTo.(type) {
		case Tensor:
			m.storage[dest] = v
			return nil
		case *dualValue:
			if tv, ok := v.Value.(Tensor); ok {
				m.storage[dest] = tv
				return nil
			}
		}
	}

	machineLogf("Have to allocate %v in register %v", instr.t, instr.writeTo)
	var tt TensorType
	var ok bool
	if tt, ok = instr.t.(TensorType); !ok {
		return errors.New("Alloc only allocates tensor types")

		// allocate a "scalar" vector
	}

	var dt Dtype
	if dt, ok = hm.Prune(tt.of).(Dtype); !ok {
		return errors.Errorf("No dtype to allocate. Type: %T", tt.of)
	}

	//TODO: runtime shape check

	t := NewTensorValue(dt, instr.s...)

	m.storage[dest] = t

	return
}

func (instr alloc) String() string {
	return fmt.Sprintf("Alloc %v\t\t%v", instr.t, instr.writeTo)
}

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

	m.storage[instr.writeTo.id] = v
	m.watchedLogf("Write To: %v", instr.writeTo)
	m.watchedLogf(m.valueFmt, m.storage[instr.writeTo.id])
	return nil
}

func (instr loadArg) String() string {
	return fmt.Sprintf("loadArg %x to %v", instr.index, instr.writeTo)
}

type execOp struct {
	op          Op
	inputTypes  hm.Types
	outputType  hm.Type
	outputShape types.Shape

	id int

	readFrom []register
	writeTo  register

	preAllocated bool
	useUnsafe    bool
}

func (instr execOp) ID() int           { return instr.id }
func (instr execOp) reads() []register { return instr.readFrom }
func (instr execOp) writes() register  { return instr.writeTo }

func newExecOp(n *Node) execOp {
	var inputTypes hm.Types
	for _, child := range n.children {
		inputTypes = append(inputTypes, child.t)
	}

	return execOp{
		op:         n.op,
		id:         n.ID(),
		inputTypes: inputTypes,
		outputType: n.t,
	}
}

func (instr execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	// Read
	m.watchedLogf("Inputs:")
	m.enterLoggingContext()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.storage[reg.id]
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLoggingContext()

	// Execute
	var v Value
	switch {
	case instr.preAllocated:
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p := m.storage[instr.writeTo.id]
			if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
				return errors.Wrapf(err, "Happened while attempting to execute %v. Node is %x. Register was: %v ", instr, instr.id, instr.writeTo.id)
			}
		} else {
			// TODO: maybe warn?
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}
	case instr.useUnsafe:
		if ud, ok := instr.op.(UnsafeDoer); ok {
			if v, err = ud.UnsafeDo(inputs...); err != nil {
				return errors.Wrap(err, "Failed to carry UnsafeDo()")
			}
		} else {
			// TODO: warn?
			if v, err = instr.op.Do(inputs...); err != nil {
				return errors.Wrap(err, opDoFail)
			}
		}
	default:
		if v, err = instr.op.Do(inputs...); err != nil {
			return errors.Wrap(err, opDoFail)
		}
	}

	m.watchedLogf("Result:")
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()
	// TODO: type and shape checks

	// Write
	dest := instr.writeTo.id
	m.storage[dest] = v
	node := m.p.g.Node(instr.id).(*Node)

	if m.trace() {
		var cloned Value
		if cloned, err = v.clone(); err != nil {
			return errors.Wrap(err, cloneFail)
		}
		node.bind(cloned)
	} else {
		node.bind(v)
	}

	// this is a gradient node then, we should also bind the value to the node's dualValue
	if node.derivOf != nil {
		for _, src := range node.derivOf {
			if src.boundTo != nil {
				dv := dvUnit0(src.boundTo)
				dv.SetDeriv(v) // important!! do NOT use node.boundTo
				src.bind(dv)
			}
		}
	}
	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()
	return nil
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
	m.b.DoWork()
	return nil
}

func (instr flushInstr) ID() int           { return -1 }
func (instr flushInstr) reads() []register { return nil }
func (instr flushInstr) writes() register  { return register{-1, CPU} }
func (instr flushInstr) String() string    { return "Do Batched BLAS" }

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
	v := m.storage[instr.readFrom.id]
	v2, err := v.clone()
	if err != nil {
		return errors.Wrap(err, cloneFail)
	}

	*instr.into = v2
	return nil
}

func (instr readInstr) String() string {
	return fmt.Sprintf("Read %v into %p", instr.readFrom, instr.into)
}
