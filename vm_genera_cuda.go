// +build cuda

package gorgonia

import (
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func (m *lispMachine) init() error {
	if err := m.prepGraph(); err != nil {
		return err
	}

	// VERY simple data analysis (even simpler than the one used in Compile)
	// using replaceWithSelf reduces the need for hashing, hence less work is required
	// However this also means that CSE won't be performed
	df := newdataflow()
	df.replaceWithSelf(m.sorted)
	df.buildIntervals(m.sorted)
	df.fixIntervalDevices(m.sorted)
	m.df = df

	if err := m.calcMemSize(); err != nil {
		log.Printf("err1")
		return err
	}

	if len(m.gpumem) == 0 {
		m.ForceCPU()
		return nil
	}

	if err := m.ExternMetadata.init(m.gpumem); err != nil {
		m.ExternMetadata.initFail()
		return err
	}
	m.loadStdLib()

	if len(m.engines) == 0 {
		m.ForceCPU()
	}
	return nil
}

func finalizeLispMachine(m *lispMachine) {
	m.ExternMetadata.cleanup()
	m.ExternMetadata.initFail()
}

func (m *lispMachine) WorkAvailable() <-chan bool {
	if m.ExternMetadata.WorkAvailable() == nil {
		return nil
	}
	return m.ExternMetadata.WorkAvailable()
}

func (m *lispMachine) calcMemSize() (err error) {
	compileLogf("calcmemsize")
	enterLogScope()
	defer leaveLogScope()
	var cpumem int64
	var gpumem []int64
	for _, n := range m.sorted {
		interv := m.df.intervals[n]
		dev := interv.result.device
		compileLogf("n: %v | %v", n, interv)

		var dt tensor.Dtype
		if dt, err = dtypeOf(n.t); err != nil {
			if n.isStmt {
				continue
			}
			return errors.Wrapf(err, "Cannot calculate memsize of n(%v)", n)
		}
		switch {
		case n.isArg():
			cpumem += calcMemSize(dt, n.Shape())
		case n.isStmt:
		default:
			// if !n.op.ReturnsPtr() {
			if dev != CPU {
				if len(gpumem) < int(dev)+1 {
					diff := int(dev) + 1 - len(gpumem)
					gpumem = append(gpumem, make([]int64, diff)...)
				}
			}

			switch dev {
			case CPU:
				cpumem += calcMemSize(dt, n.Shape())
			default:
				compileLogf("n: %v. AddedDEF", n)
				gpumem[int(dev)] += 4 * calcMemSize(dt, n.Shape())
			}
			// }
		}
	}

	m.cpumem = cpumem
	m.gpumem = gpumem
	return nil
}

func (m *lispMachine) execDevTrans(op devTrans, n *Node, children Nodes) (err error) {
	child := children[0]
	m.logf("DevTrans: %v | %v | %v", op, n.boundTo, child.boundTo)

	var dv *dualValue
	var cv, cd, v, d Value
	if child.boundTo != nil {
		var ok bool
		if dv, ok = child.boundTo.(*dualValue); ok {
			cv = dv.Value
			cd = dv.d
		} else {
			cv = child.boundTo
		}
	} else {
		err = errors.Errorf("Cannot execute transfer when there is no value in child")
		return
	}

	var synchronous bool
	if op.to == CPU && op.from != CPU {
		synchronous = true
	}

	if v, err = m.Transfer(op.to, op.from, cv, false); err != nil {
		return
	}

	if cd != nil {
		if d, err = m.Transfer(op.to, op.from, cd, false); err != nil {
			return
		}
	} else {
		var mem tensor.Memory
		if mem, err = m.Get(op.to, calcMemSize(cv.Dtype(), child.shape)); err != nil {
			return
		}
		if _, err = makeValueFromMem(child.t, child.shape, mem); err != nil {
			return
		}
	}

	if synchronous {
		m.Signal()
	}

	dv = new(dualValue)
	dv.Value = v
	dv.d = d
	n.boundTo = dv

	return nil
}

// loads the standardlib
func (m *lispMachine) loadStdLib() {
	if cudaStdLib == nil {
		return
	}

	for name, data := range cudaStdLib {
		funcs, ok := cudaStdFuncs[name]
		if !ok {
			cudaLogf("No funcs for module %q", name)
			// panic("WTF")
			continue
		}
		for i := range m.engines {
			e := &m.engines[i]
			if err := e.LoadCUDAFunc(name, data, funcs); err != nil {
				panic(err)
			}

		}
	}
}

// ForceCPU forces the lispMachine to have the nodes run on the CPU
func (m *lispMachine) ForceCPU() {
	m.cleanup()
	m.initFail()
	m.df = nil

	for _, n := range m.sorted {
		n.dataOn = CPU
	}

	// remove devTrans if any
	for i := 0; i < len(m.sorted); i++ {
		n := m.sorted[i]
		if _, ok := n.op.(devTrans); ok {
			copy(m.sorted[i:], m.sorted[i+1:])
			m.sorted = m.sorted[:len(m.sorted)-1]
			i--
		}
	}
}
