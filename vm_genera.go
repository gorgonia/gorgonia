package gorgonia

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"strings"

	"github.com/pkg/errors"
)

type lispMachine struct {
	g *ExprGraph
	q []adInstr // a to-do list of differentiation instructions

	// state stuff, to allow continuation
	sorted Nodes
	fwd    int
	bwd    int

	// logging stuff
	watchlist Nodes
	logger    *log.Logger
	buf       *bytes.Buffer
	valueFmt  string
	tabcount  int
	logFlags  byte

	runFlags     byte // supposed to go into state stuff.  Placed here for better compacting of struct
	checkedRoots bool // supposed to go into state stuff.
}

func NewLispMachine(g *ExprGraph, opts ...VMOpt) *lispMachine {
	runFlags := (byte(0) | (byte(1) << fwdOnly)) | (1 << bwdOnly) // run fwd and backwards
	m := &lispMachine{
		g:        g,
		fwd:      -1,
		bwd:      -1,
		valueFmt: "%3.3f",
		logFlags: 0x0,      // log nothing
		runFlags: runFlags, // run only fwd and bwd
	}

	for _, opt := range opts {
		opt(m)
	}
	return m
}

func (m *lispMachine) logBwd() bool { return (m.logFlags>>bwdOnly)&byte(1) == 1 }
func (m *lispMachine) doLogBwd()    { m.logFlags |= byte(1) << bwdOnly }
func (m *lispMachine) dontLogBwd()  { m.logFlags &= (^(byte(1) << bwdOnly)) }
func (m *lispMachine) runBwd() bool { return m.runFlags>>bwdOnly&byte(1) == 1 }
func (m *lispMachine) doExecBwd()   { m.runFlags |= byte(1) << bwdOnly }
func (m *lispMachine) dontExecBwd() { m.runFlags &= (^(byte(1) << bwdOnly)) }

func (m *lispMachine) logFwd() bool { return (m.logFlags>>fwdOnly)&byte(1) == 1 }
func (m *lispMachine) doLogFwd()    { m.logFlags |= byte(1) << fwdOnly }
func (m *lispMachine) dontLogFwd()  { m.logFlags &= (^(byte(1) << fwdOnly)) }
func (m *lispMachine) runFwd() bool { return m.runFlags>>fwdOnly&byte(1) == 1 }
func (m *lispMachine) doExecFwd()   { m.runFlags |= byte(1) << fwdOnly }
func (m *lispMachine) dontExecFwd() { m.runFlags &= (^(byte(1) << fwdOnly)) }

func (m *lispMachine) watchNaN() bool { return (m.runFlags>>watchNaN)&byte(1) == 1 }
func (m *lispMachine) doWatchNaN()    { m.runFlags |= byte(1) << watchNaN }
func (m *lispMachine) dontWatchNaN()  { m.runFlags &= (^(byte(1) << watchNaN)) }

func (m *lispMachine) watchInf() bool { return (m.runFlags>>watchInf)&byte(1) == 1 }
func (m *lispMachine) doWatchInf()    { m.runFlags |= byte(1) << watchInf }
func (m *lispMachine) dontWatchInf()  { m.runFlags &= (^(byte(1) << watchInf)) }

func (m *lispMachine) watchAll() bool { return (m.logFlags>>watchAll)&byte(1) == 1 }
func (m *lispMachine) doWatchAll()    { m.logFlags |= (byte(1) << watchAll) }
func (m *lispMachine) dontWatchAll()  { m.logFlags &= (^(byte(1) << watchAll)) }

func (m *lispMachine) dealloc() bool { return (m.runFlags>>allocVals)&byte(1) == 1 }
func (m *lispMachine) doDealloc()    { m.runFlags |= byte(1) << allocVals }
func (m *lispMachine) dontDealloc()  { m.runFlags &= (^(byte(1) << allocVals)) }

// check roots only applies if you want to run a backprop as well
func (m *lispMachine) checkRoots() (err error) {
	if !m.checkedRoots && m.runBwd() {
		machineLogf("Checking if provided graph is sensible")
		machineLogf("roots: %v", m.g.Roots())
		for _, root := range m.g.Roots() {
			if !root.IsScalar() && !root.isStmt {
				err = NewError(AutoDiffError, "Expected cost to be a scalar. Got %v with shape %v instead", root, root.Shape())
				ioutil.WriteFile("err.dot", []byte(root.RestrictedToDot(2, 10)), 0644)
				return
			}
		}
	}
	return
}

func (m *lispMachine) prepGraph() (err error) {
	if m.sorted == nil {
		if m.sorted, err = Sort(m.g); err != nil {
			err = errors.Wrap(err, sortFail)
			return
		}

		m.fwd = len(m.sorted) - 1
	}
	return
}

func (m *lispMachine) forward() (err error) {
	if m.fwd < 0 {
		return nil // or err?
	}

	n := m.sorted[m.fwd]
	m.watchedLogf("n: %v (%x)", n, n.Hashcode())
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	if n.isInput() {
		machineLogf("Unit() on input node")
		if err = n.bind(dvUnit(n.boundTo)); err != nil {
			return
		}
		m.watchedLogf(m.valueFmt, n.boundTo)
		return
	}

	// other wise it's time to execute the op
	m.watchedLogf("execute Op")
	op := n.op
	var output *dualValue

	inputs := make([]*dualValue, len(n.children))
	for i, child := range n.children {
		dv := child.boundTo.(*dualValue)
		inputs[i] = dv
	}

	m.watchedLogf("Before:")
	m.watchedLogf(m.valueFmt, n.boundTo)

	switch {
	case (m.g.roots.Contains(n) || n.isRoot()) && !n.isStmt:
		machineLogf("Applying op %v to root", op)
		if n.boundTo == nil {
			if output, err = dvBindVar(op, inputs); err != nil {
				return
			}
			if err = n.bind(output); err != nil {
				return
			}
		} else {
			dv := n.boundTo.(*dualValue)
			if err = dvBindVar0(op, dv, inputs); err != nil {
				return
			}
		}

	case n.isStmt:
		machineLogf("ReadOp: %v ", op)
		switch ot := n.op.(type) {
		case readOp:
			childVal := n.children[0].boundTo
			if dv, ok := childVal.(*dualValue); ok {
				*ot.into = dv.Value
			} else {
				*ot.into = childVal
			}
		}

	case n.boundTo == nil:
		machineLogf("Fresh, unencountered node, so dvBind(%v)", op)
		machineLogf("Inputs")
		enterLoggingContext()
		for i, in := range inputs {
			if inT, ok := in.Value.(Tensor); ok {
				machineLogf("%d; %v", i, inT.Tensor.Info())
			}
		}
		leaveLoggingContext()
		if output, err = dvBind(op, inputs); err != nil {
			err = errors.Wrapf(err, execFail, op)
			return
		}

		if err = n.bind(output); err != nil {
			return
		}

	default:
		machineLogf("bind(%v) with as much reuse as possible", op)
		// reuse as much as possible
		output := dvUnit(n.boundTo)
		if err = n.bind(output); err != nil {
			return
		}

		err = dvBind0(op, output, inputs)
		if et, ok := errors.Cause(err).(errorTyper); ok {
			if et.ErrorType() != AutoDiffError {
				err = errors.Wrapf(err, execFail, op)
				return
			}
			err = nil
		} else if err != nil {
			err = errors.Wrapf(err, execFail, op)
			return
		}
	}
	m.watchedLogf("After:")
	m.watchedLogf(m.valueFmt, n.boundTo)

	if aop, ok := op.(AdOp); ok && m.runBwd() {
		instr := adInstr{
			AdOp: aop,

			inputs: n.children,
			output: n,
		}
		m.q = append(m.q, instr)
	}

	if m.watchNaN() && !n.isStmt {
		if hasNaN(n.boundTo) {
			err = newValueErr(n, "NaN found in value")
			return
		}
	}

	return
}

func (m *lispMachine) backward() (err error) {
	if m.bwd < 0 {
		return NewError(RuntimeError, "no backprop queue")
	}

	instr := m.q[m.bwd]
	m.watchedLogf("Differentiating op %v. Output: %v (%x)", instr, instr.output, instr.output.Hashcode())
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	m.watchedLogf("Inputs: %v", instr.inputs)
	m.enterLoggingContext()
	for _, in := range instr.inputs {
		m.watchedLogf(m.valueFmt, in.boundTo.(*dualValue).d)
	}
	m.leaveLoggingContext()

	// actual differentiation
	if err = instr.do(); err != nil {
		err = errors.Wrapf(err, autodiffFail, instr.AdOp)
		return
	}

	m.watchedLogf("After:")
	m.enterLoggingContext()
	for _, in := range instr.inputs {
		m.watchedLogf(m.valueFmt, in.boundTo.(*dualValue).d)
	}

	m.leaveLoggingContext()

	if m.watchNaN() {
		if hasNaN(instr.output.boundTo) {
			err = newValueErr(instr.output, "NaN found in value")
			return
		}

		for _, in := range instr.inputs {
			if hasNaN(in.boundTo) {
				err = newValueErr(in, "NaN found in value")
				return
			}
		}
	}
	return
}

func (m *lispMachine) RunAll() (err error) {
	if err = m.prepGraph(); err != nil {
		return
	}

	if err = m.checkRoots(); err != nil {
		return
	}

	if !m.runFwd() {
		goto backward
	}

	for err = nil; err == nil && m.fwd >= 0; m.fwd-- {
		err = m.forward()
	}

	if err != nil {
		return
	}

backward:
	if !m.runBwd() {
		return nil
	}

	if m.bwd < 0 {
		m.bwd = len(m.q) - 1
	}

	for err = nil; err == nil && m.bwd >= 0; m.bwd-- {
		err = m.backward()
	}
	return
}

func (m *lispMachine) Free() {
	if m.dealloc() {
		for _, n := range m.sorted {
			m.logf("dealloc n; %v %x %p", n, n.Hashcode(), n)
			if !n.isInput() {
				n.unbind()
				m.logf("OK")
			}
		}
	}
}

func (m *lispMachine) Reset() {
	m.fwd = len(m.sorted)
	m.bwd = len(m.q)
}

func (m *lispMachine) LastRun() (n *Node, backprop bool) {
	if m.fwd < 0 && m.runBwd() {
		goto backward
	} else if !m.runBwd() {
		n = m.sorted[0] // last to run
		return
	} else {
		n = m.sorted[m.fwd]
		return
	}

backward:
	backprop = true
	if m.bwd < 0 {
		n = m.q[0].output
		return
	}
	n = m.q[m.bwd].output
	return
}

func (m *lispMachine) watchedLogf(format string, attrs ...interface{}) {
	if !m.logFwd() {
		goto backwards
	}

	if m.fwd >= 0 {
		n := m.sorted[m.fwd]
		if m.watchlist.Contains(n) || m.watchAll() || DEBUG {
			m.logf(format, attrs...)
		}
		return
	}

backwards:
	if !m.logBwd() {
		return
	}

	if m.bwd >= 0 {
		instr := m.q[m.bwd]
		write := m.watchlist.Contains(instr.output)
		if !write {
			for _, in := range instr.inputs {
				if m.watchlist.Contains(in) {
					write = true
					break
				}
			}
		}

		if write || m.watchAll() || DEBUG {
			m.logf(format, attrs...)
		}
	}
}

func (m *lispMachine) logf(format string, attrs ...interface{}) {
	switch {
	case machineDev, autodiffDev:
		if machineDev {
			machineLogf(format, attrs...)
		} else {
			autodiffLogf(format, attrs...)
		}

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

func (m *lispMachine) enterLoggingContext() {
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

func (m *lispMachine) leaveLoggingContext() {
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

// adInstr is an autodifferentiation instruction
type adInstr struct {
	AdOp

	inputs Nodes
	output *Node
}

func (instr adInstr) do() error {
	return instr.AdOp.DoDiff(instr.inputs, instr.output)
}
