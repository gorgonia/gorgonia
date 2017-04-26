package gorgonia

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"runtime"
	"strings"

	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
)

type lispMachine struct {
	*ExternMetadata
	g *ExprGraph
	q []adInstr // a to-do list of differentiation instructions

	// device stuff
	cpumem int64
	gpumem []int64 // gpumem is indexed by gpuid

	// state stuff, to allow continuation
	sorted Nodes
	df     *dataflow
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

// NewLispMachine creates a VM that executes the graph as it is traversed. Depending on the VMOpts passed in
// this VM is also capable of performing automatic differentiation.
func NewLispMachine(g *ExprGraph, opts ...VMOpt) *lispMachine {
	runFlags := (byte(0) | (byte(1) << fwdOnly)) | (1 << bwdOnly) // run fwd and backwards
	m := &lispMachine{
		ExternMetadata: new(ExternMetadata),
		g:              g,
		fwd:            -1,
		bwd:            -1,
		valueFmt:       "%3.3f",
		logFlags:       0x0,      // log nothing
		runFlags:       runFlags, // run only fwd and bwd
	}

	for _, opt := range opts {
		opt(m)
	}
	if err := m.init(); err != nil {
		panic(err)
	}
	runtime.SetFinalizer(m, finalizeLispMachine)
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

func (m *lispMachine) setRootGrad() bool    { return (m.runFlags>>spare3)&byte(1) == 1 }
func (m *lispMachine) allowSetRootGrad()    { m.runFlags |= byte(1) << spare3 }
func (m *lispMachine) disallowSetRootGrad() { m.runFlags &= (^(byte(1) << spare3)) }

func (m *lispMachine) Reset() {
	m.fwd = len(m.sorted) - 1
	m.bwd = len(m.q) - 1
}

func (m *lispMachine) RunAll() (err error) {
	if err = m.checkRoots(); err != nil {
		return errors.Wrap(err, "Could not checkRoots()")
	}

	defer func() {
		m.q = nil // this needs to be nil'd or else there would still be references to m. Then there won't be any garbage being collected
	}()

	workAvailable := m.ExternMetadata.WorkAvailable()
	syncChan := m.ExternMetadata.Sync()
	errChan := make(chan error)
	doneChan := make(chan struct{})

	go m.runall(errChan, doneChan)
	for {
		select {
		case synchronous := <-workAvailable:
			err := m.ExternMetadata.DoWork()
			if err != nil {
				return err
			}
			if synchronous {
				syncChan <- struct{}{}
			}
		case err := <-errChan:
			if m.fwd < len(m.sorted) {
				return errors.Wrapf(err, "Running Node: %v", m.sorted[m.fwd])
			}
			return errors.Wrap(err, "RunAll")
		case <-doneChan:
			err := m.ExternMetadata.DoWork()
			if err != nil {
				return err
			}
			return nil
		default:
		}
	}

	return nil
}

func (m *lispMachine) RunForwards() (err error) {
	if err = m.checkRoots(); err != nil {
		return errors.Wrap(err, "Could not check roots")
	}

	if !m.runFwd() {
		return errors.New("Cannot run forwards")
	}

	for err = nil; err == nil && m.fwd < len(m.sorted); m.fwd++ {
		err = m.forward()
	}
	return
}

func (m *lispMachine) RunBackwards() (err error) {
	if err = m.checkRoots(); err != nil {
		return errors.Wrap(err, "Could not check roots")
	}

	if !m.runBwd() || len(m.q) == 0 {
		return errors.New("Cannot run backwards")
	}
	m.bwd = len(m.q) - 1

	for err = nil; err == nil && m.bwd >= 0; m.bwd-- {
		err = m.backward()
	}
	return
}

func (m *lispMachine) UnbindAll() {
	// if m.dealloc() {
	for _, n := range m.sorted {
		m.logf("dealloc n; %v %x %p", n, n.Hashcode(), n)
		if !n.isInput() {
			n.unbind()
			m.logf("OK")
		}
	}
	// }
}

func (m *lispMachine) UnbindOnDevice() {
	for _, n := range m.sorted {
		if !n.isInput() && n.dataOn != CPU {
			n.unbind()
		}
	}
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

// check roots only applies if you want to run a backprop as well
func (m *lispMachine) checkRoots() (err error) {
	if !m.checkedRoots && m.runBwd() {
		machineLogf("Checking if provided graph is sensible")
		m.logf("roots: %v", m.g.Roots())
		for _, root := range m.g.Roots() {
			switch {
			case m.setRootGrad() && !root.isStmt:
				// check root's value
				if _, ok := root.boundTo.(*dualValue); !ok {
					err = errors.Errorf("Expected root %v to have a boundTo of a dualValue", root)
					return
				}
			case !m.setRootGrad() && !root.IsScalar() && !root.isStmt:
				err = errors.Errorf("Expected cost to be a scalar. Got %v with shape %v instead", root, root.Shape())
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
			return errors.Wrap(err, sortFail)
		}
		reverseNodes(m.sorted)

		m.fwd = 0
	}
	return
}

func (m *lispMachine) runall(errChan chan error, doneChan chan struct{}) {
	var err error
	if !m.runFwd() {
		goto backward
	}

	for err = nil; err == nil && m.fwd < len(m.sorted); m.fwd++ {
		err = m.forward()
	}

	if err != nil {
		logf("Sending error")
		errChan <- err
	}

	// send a synchronous signal, do all (if any) CUDA work before continuing with backprop
	m.Signal()

backward:
	if !m.runBwd() {
		doneChan <- struct{}{}
		return
	}

	if m.bwd < 0 {
		m.bwd = len(m.q) - 1
	}

	for err = nil; err == nil && m.bwd >= 0; m.bwd-- {
		err = m.backward()
	}
	if err != nil {
		errChan <- err
	}
	doneChan <- struct{}{}
}

func (m *lispMachine) forward() (err error) {
	if m.fwd < 0 {
		return nil // or err?
	}

	n := m.sorted[m.fwd]
	m.watchedLogf("n: %v (%x)", n, n.Hashcode())
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	if !n.isStmt {
		switch {
		case n.isArg():
			machineLogf("Unit() on input node")
			if err = n.bind(dvUnit(n.boundTo)); err != nil {
				return errors.Wrap(err, bindFail)
			}
			return
		case n.isRandom():
			machineLogf("binding value of random node")

			var v Value
			if v, err = n.op.Do(); err != nil {
				return errors.Wrapf(err, execFail, n.op, n)
			}

			// we wrap it in a dualValue, but don't allocate anything for the d
			if err = n.bind(dvUnit0(v)); err != nil {
				return errors.Wrap(err, bindFail)
			}
			return
		default:
			// do nothihng
		}
		m.watchedLogf(m.valueFmt, n.boundTo)
	}

	// other wise it's time to execute the op
	m.logf("execute Op")

	dev := n.dataOn
	op := NewExternalOp(n.op, ExecutionContext{m, dev}, nil)

	// m.watchedLogf("Result of execution of this node would reside in %v", dev)
	var output *dualValue

	inputs := make([]*dualValue, len(n.children))
	children := n.children
	if m.df != nil {
		var ok bool
		if children, ok = m.df.devTransChildren[n]; !ok {
			children = n.children
		}
	}

	m.enterLoggingContext()
	for i, child := range children {
		dv, ok := child.boundTo.(*dualValue)
		if !ok {
			log.Printf("%t, %t", m.g.Has(n), m.g.Has(child))
			log.Printf("%v", n)
			log.Printf("%v", child)
			ioutil.WriteFile("FAIL.dot", []byte(n.RestrictedToDot(1, 3)), 0644)
			panic(fmt.Sprintf("Expected child to have bound Value. Got %v instead", child.boundTo))
		}
		inputs[i] = dv
	}
	m.leaveLoggingContext()

	m.watchedLogf("Before:")
	m.watchedLogf(m.valueFmt, n.boundTo)

	switch {
	case (m.g.roots.Contains(n) || n.isRoot()) && !n.isStmt:
		machineLogf("Applying op %v to root", op)
		if n.boundTo == nil {
			machineLogf("dvBindVar")
			m.logf("dvBindVar")
			if output, err = dvBindVar(op, inputs); err != nil {
				return errors.Wrapf(err, execFail, op, n)
			}
			if err = n.bind(output); err != nil {
				return errors.Wrap(err, bindFail)
			}
		} else {
			machineLogf("dvBindVar0")
			m.logf("dvBindVar0")
			dv, ok := n.boundTo.(*dualValue)
			if !ok {
				panic(fmt.Sprintf("n not dual value %v", n))
			}
			if err = dvBindVar0(op, dv, inputs); err != nil {
				return errors.Wrapf(err, execFail, op, n)
			}
		}

	case n.isStmt:
		switch ot := n.op.(type) {
		case readOp:
			machineLogf("ReadOp: %v ", op)
			child := children[0]
			childVal := child.boundTo
			if child.Device() != CPU {
				m.Signal() // get work to be done first

				if dv, ok := n.children[0].boundTo.(*dualValue); ok {
					*ot.into = dv.Value
				} else {
					*ot.into = childVal
				}

			} else {
				if dv, ok := childVal.(*dualValue); ok {
					*ot.into = dv.Value
				} else {
					*ot.into = childVal
				}
			}

		case devTrans:
			// this case is unreachable in non CUDA builds
			if err = m.execDevTrans(ot, n, children); err != nil {
				return errors.Wrapf(err, "Cannot perform device transfer %v", ot)
			}
		}

	case n.boundTo == nil:
		m.watchedLogf("Fresh, unencountered node, so dvBind(%v)", op)
		machineLogf("Inputs")
		enterLoggingContext()
		for i, in := range inputs {
			if inT, ok := in.Value.(tensor.Tensor); ok {
				machineLogf("%d; %v", i, inT.Info())
			}
		}
		leaveLoggingContext()
		if output, err = dvBind(op, inputs); err != nil {
			return errors.Wrapf(err, execFail, op, n)
		}

		if err = n.bind(output); err != nil {
			return errors.Wrap(err, bindFail)
		}

	default:
		m.logf("bind(%v) with as much reuse as possible", op)
		// reuse as much as possible
		output := dvUnit(n.boundTo)
		if err = n.bind(output); err != nil {
			return errors.Wrap(err, bindFail)
		}

		err = dvBind0(op, output, inputs)
		if _, ok := errors.Cause(err).(AutoDiffError); ok {
			err = nil
		} else if err != nil {
			return errors.Wrapf(err, execFail, op, n)
		}
	}
	m.watchedLogf("After:")
	m.watchedLogf(m.valueFmt, n.boundTo)

	if aop, ok := op.Op.(ADOp); ok && m.runBwd() {
		instr := adInstr{
			ADOp: aop,
			ctx:  op.ExecutionContext,

			inputs: n.children, // this is correct.
			output: n,
		}
		m.q = append(m.q, instr)
	}
	m.watchedLogf("Added to Queue")

	if m.watchNaN() && !n.isStmt {
		if hasNaN(n.boundTo) {
			return errors.New("NaN found in value")
		}
	}

	return
}

func (m *lispMachine) backward() (err error) {
	if m.bwd < 0 {
		return errors.New("no backprop queue")
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
		return errors.Wrapf(err, autodiffFail, instr.ADOp)
	}

	m.watchedLogf("After:")
	m.enterLoggingContext()
	for _, in := range instr.inputs {
		m.watchedLogf(m.valueFmt, in.boundTo.(*dualValue).d)
	}

	m.leaveLoggingContext()

	if m.watchNaN() {
		if hasNaN(instr.output.boundTo) {
			return errors.New("NaN found in value")
		}

		for _, in := range instr.inputs {
			if hasNaN(in.boundTo) {
				return errors.New("NaN found in value")
			}
		}
	}
	return
}

func (m *lispMachine) watchedLogf(format string, attrs ...interface{}) {
	if !m.logFwd() && !DEBUG {
		goto backwards
	}

	if m.fwd >= 0 && m.fwd < len(m.sorted) {
		n := m.sorted[m.fwd]
		if m.watchlist.Contains(n) || m.watchAll() {
			m.logf(format, attrs...)
		}
		return
	}

backwards:
	if !m.logBwd() && !DEBUG {
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

func (m *lispMachine) leaveLoggingContext() {
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

// adInstr is an autodifferentiation instruction
type adInstr struct {
	ADOp
	ctx ExecutionContext

	inputs Nodes
	output *Node
}

func (instr adInstr) do() error {
	if instr.output.dataOn != CPU {
		for _, in := range instr.inputs {
			if in.dataOn == CPU {
				// ensure everything gets executed in the GPU first
				instr.ctx.Signal()
				break
			}
		}
	}
	err := instr.ADOp.DoDiff(instr.ctx, instr.inputs, instr.output)
	// logf("INPUTS:")
	// for _, in := range instr.inputs {
	// 	logf("%v\n", in.boundTo.(*dualValue).d)
	// }
	return err
}
