// +build cuda

package gorgonia

import (
	"fmt"

	"github.com/chewxy/cu"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
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
	m.sorted = df.insertDeviceInstr(m.sorted)
	df.buildIntervals(m.sorted)
	df.fixIntervalDevices(m.sorted)
	m.df = df

	if err := m.calcMemSize(); err != nil {
		return err
	}

	cudaLogf("%v", m.f)
	funcs := make([]string, 0, len(m.ExternMetadata.f))
	for fn := range m.f {
		funcs = append(funcs, fn)
	}
	m.ExternMetadata.init(m.gpumem)
	m.loadStdLib()
	return nil
}

func finalizeLispMachine(m *lispMachine) {
	cudaLogf("Finalizing lispMachine %p", m)
	for i, c := range m.c {
		cu.SetCurrent(c.Context)
		for _, v := range m.m {
			mod := v[i]
			cu.Unload(mod)
		}
		cu.DestroyContext(&c.Context)
	}
	m.Cleanup()
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
			return errors.Wrapf(err, "Cannop calulate memsize of n(%v)", n)
		}
		switch {
		case n.isArg():
			cpumem += calcMemSize(dt, n.Shape())
		case n.isStmt:
			if trans, ok := n.op.(devTrans); ok {
				switch trans.to {
				case CPU:
					cpumem += calcMemSize(dt, n.Shape())
				default:
					if dev != CPU {
						if len(gpumem) < int(dev)+1 {
							diff := int(dev) + 1 - len(gpumem)
							gpumem = append(gpumem, make([]int64, diff)...)
						}
					}

					compileLogf("n: %v. Added Stmt", n)
					gpumem[int(trans.to)] += 2 * calcMemSize(dt, n.Shape())
				}
			}
		default:
			if !n.op.ReturnsPtr() {
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
					gpumem[int(dev)] += 2 * calcMemSize(dt, n.Shape())
				}
			}
		}
	}

	m.cpumem = cpumem
	m.gpumem = gpumem
	return nil
}

func (m *lispMachine) execDevTrans(op devTrans, n *Node, children Nodes) (err error) {
	m.watchedLogf("DevTrans: %v |%v", op, n.boundTo)
	child := children[0]
	var dt tensor.Dtype
	if dt, err = dtypeOf(child.t); err != nil {
		return errors.Wrapf(err, "Unable to get dtype of %v while executing devTrans", child.t)
	}

	var dv *dualValue
	if n.boundTo != nil {
		var ok bool
		if dv, ok = n.boundTo.(*dualValue); !ok {

		}
	}

	switch {
	case op.from != CPU && op.to == CPU:
		var v, d Value
		if v, err = makeValue(child.t, child.Shape()); err != nil {
			return errors.Wrapf(err, "Unable to make value of %v and %v", child.t, child.Shape())
		}
		if d, err = makeValue(child.t, child.Shape()); err != nil {
			return errors.Wrapf(err, "Unable to make value of %v and %v", child.t, child.Shape())
		}

		dv = new(dualValue)
		dv.Value = v
		dv.d = d
		n.boundTo = dv

	case op.from == CPU && op.to != CPU:
		memsize := calcMemSize(dt, child.Shape())
		var memV, memD Memory
		if memV, err = m.Get(op.to, memsize); err != nil {
			return errors.Wrapf(err, "Unable to allocate %v bytes from %v", memsize, op.to)
		}

		if memD, err = m.Get(op.to, memsize); err != nil {
			return errors.Wrapf(err, "Unable to allocate %v bytes from %v", memsize, op.to)
		}
		m.logf("Allocated 0x%x", memD.Uintptr())

		var v, d Value
		if v, err = makeValueFromMem(child.t, child.Shape(), memV); err != nil {
			return errors.Wrapf(err, "Unable to make value of %v and %v from memory", child.t, child.Shape())
		}
		if d, err = makeValueFromMem(child.t, child.Shape(), memD); err != nil {
			return errors.Wrapf(err, "Unable to make value of %v and %v from memory", child.t, child.Shape())
		}
		m.logf("V: \n%v|%v", v, v.Shape())

		cv := child.Value()
		ctx := m.Contexts()[op.to]
		ctx.MemcpyHtoD(cu.DevicePtr(v.Uintptr()), cv.Pointer(), memsize)

		dv = new(dualValue)
		dv.Value = v
		dv.d = d
		n.boundTo = dv
	default:
		logf("No op")
	}
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
			continue
		}
		if err := m.LoadCUDAFunc(name, data, funcs); err != nil {
			cudaLogf("Unable to load %q.: %v", name, err)
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
		if err = cu.SetCurrent(c.Context); err != nil {
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
		if err = cu.SetCurrent(m.c[0].Context); err != nil {
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
