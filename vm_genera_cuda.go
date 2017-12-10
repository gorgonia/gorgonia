// +build cuda

package gorgonia

import (
	"fmt"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
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
		return err
	}

	if len(m.gpumem) == 0 {
		m.ForceCPU()
		return nil
	}

	cudaLogf("%v", m.f)
	funcs := make([]string, 0, len(m.ExternMetadata.f))
	for fn := range m.f {
		funcs = append(funcs, fn)
	}
	m.ExternMetadata.init(m.gpumem)
	m.loadStdLib()

	if len(m.Functions()) == 0 {
		m.ForceCPU()
	}
	return nil
}

func finalizeLispMachine(m *lispMachine) {
	m.cleanup()
	m.initFail()
}

func (m *lispMachine) WorkAvailable() <-chan bool {
	if m.ExternMetadata.WorkAvailable() == nil {
		return nil
	}
	return m.ExternMetadata.WorkAvailable()
}

func (m *lispMachine) calcMemSize() (err error) {
	compileLogf("calcmemsize")
	enterLoggingContext()
	defer leaveLoggingContext()
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
			return errors.Wrapf(err, "Cannot calulate memsize of n(%v)", n)
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
		if cd, err = makeValueFromMem(child.t, child.shape, mem); err != nil {
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
		if err := m.LoadCUDAFunc(name, data, funcs); err != nil {
			cudaLogf("Unable to load %q.: %v", name, err)
			// panic(err)
		}
	}
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the machine.
//
// The convention is to have one function per module, sharing the same name.
func (m *lispMachine) LoadCUDAFunc(moduleName, data string, funcs []string) (err error) {
	if len(m.c) == 0 {
		return nil
	}

	mods := make([]cu.Module, len(m.c))
	fns := make(map[string][]cu.Function)
	for i, c := range m.c {
		if err = cu.SetCurrentContext(c.Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current context when loading module %q at context %d", moduleName, i)
			return
		}

		var mod cu.Module
		if mod, err = cu.LoadData(data); err != nil {
			err = errors.Wrapf(err, "Failed to load module %q data for %dth context %x", moduleName, i, c)
			return
		}

		var fs []cu.Function
		for _, name := range funcs {
			var ok bool
			if fs, ok = fns[name]; !ok {
				fs = make([]cu.Function, len(m.c))
			}

			var fn cu.Function
			if fn, err = mod.Function(name); err != nil {
				err = errors.Wrapf(err, "Unable to get function %q in %dth context %x", name, i, c)
				return
			}
			fs[i] = fn
			fns[name] = fs
		}

		mods[i] = mod
	}

	// set the first to current
	if len(m.c) > 0 {
		if err = cu.SetCurrentContext(m.c[0].Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current")
			return
		}
	}

	m.m[moduleName] = mods
	for _, name := range funcs {
		fqn := fmt.Sprintf("%v.%v", moduleName, name)
		m.f[fqn] = fns[name]
	}

	cudaLogf("Loaded %q", moduleName)
	return nil
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
