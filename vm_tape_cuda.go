// +build cuda

package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

func finalizeTapeMachine(m *tapeMachine) {
	cudaLogf("Finalizing tape machine %p", m)
	m.cleanup()
	m.initFail() // not really a failure. Just call to detroy all the contexts and shit
}

func (m *tapeMachine) init() {
	var initCUDA bool
	cudaLogf("instructions %v", len(m.p.instructions))
	for _, instr := range m.p.instructions {
		if eo, ok := instr.(*execOp); ok {
			if _, ok := eo.op.(CUDADoer); ok {
				initCUDA = true
				break
			}
		}
	}

	// don't bother initializing contexts if no instructions were CUDA based
	if !initCUDA {
		cudaLogf("No CUDA ops")
		return
	}

	if err := m.ExternMetadata.init(m.p.gpumem); err != nil {
		m.ExternMetadata.initFail()
		panic(err)
	}
	m.loadStdLib()

}

// loads the standardlib
func (m *tapeMachine) loadStdLib() {
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

func (m *tapeMachine) getEngine(dev Device) tensor.Engine {
	if dev == CPU {
		return m.Engine
	}
	return &m.Engines()[int(dev)]
}

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLogScope()
	defer m.leaveLogScope()

	enterLogScope()
	defer leaveLogScope()

	m.watchedLogf("Inputs:")
	m.enterLogScope()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.getValue(reg)
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v.Uintptr())
	}
	m.leaveLogScope()

	toDev := instr.writeTo.device
	var v Value
	switch op := instr.op.(type) {
	case CUDADoer:
		prealloc := m.getValue(instr.writeTo)
		if v, err = op.CUDADo(m, toDev, prealloc, inputs...); err != nil {
			return errors.Wrapf(err, "Happened while attempting to use CUDA to execute %v. Node is %x. Register was %v", instr, instr.id, instr.writeTo.id)
		}
		e := &m.Engines()[int(toDev)]
		setEngine(v, e)
	case CLDoer:
	default:
		switch {
		case instr.preAllocated:
			if pd, ok := instr.op.(UsePreallocDoer); ok {
				p := m.cpumem[instr.writeTo.id]
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
		setEngine(v, m.Engine)

	}
	m.watchedLogf("Result E:")
	m.enterLogScope()
	if vt, ok := v.(tensor.Tensor); ok {
		m.watchedLogf("%x | %T", v.Uintptr(), vt.Engine())
	} else {
		m.watchedLogf("%x", v.Uintptr())
	}
	m.leaveLogScope()
	// TODO: type and shape checks

	// Write
	m.writeValue(instr.writeTo, v)
	node := m.p.g.Node(instr.id).(*Node)

	if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
		m.Signal()
		if err = node.bindCopy(v); err != nil {
			return errors.Wrapf(err, "TraceExec failed to bind copy")
		}
	} else {
		node.bind(v)
	}

	// this is a gradient node then, we should also bind the value to the node's dualValue
	if m.bindDV() && node.derivOf != nil {
		for _, src := range node.derivOf {
			if len(m.bindNodesDV) > 0 && !m.bindNodesDV.Contains(src) {
				continue
			}

			if src.boundTo != nil {
				dv := dvUnit(src.boundTo)
				cudaLogf("dv.d 0x%x v 0x%x | writeTo: %v", dv.d.Uintptr(), v.Uintptr(), instr.writeTo)
				dev := instr.writeTo.device
				add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))
				switch dev {
				case CPU:
					if d, err := add.UnsafeDo(dv.d, v); err == nil {
						dv.SetDeriv(d)
						src.bind(dv)
					} else {
						return err
					}
				default:
					// temporarily allocate a valu
					ctx := m.Contexts()[int(dev)]

					dt := dv.d.Dtype()
					shp := dv.d.Shape()
					memsize := calcMemSize(dt, shp)

					var mem tensor.Memory
					if mem, err = m.Get(dev, memsize); err != nil {
						return errors.Wrapf(err, "Unable to allocate %v bytes from %v", memsize, dev)
					}

					var d Value
					if d, err = makeValueFromMem(dt, shp, mem); err != nil {
						return
					}

					// copy dv.d to d
					ctx.MemcpyHtoD(mem.(cu.DevicePtr), dv.d.Pointer(), memsize)

					// perform  the op
					if _, err = add.CUDADo(m, dev, d, d, v); err != nil {
						return
					}
					// copy the value back into dv.d
					ctx.MemcpyDtoH(dv.d.Pointer(), mem.(cu.DevicePtr), memsize)
					m.Put(dev, mem, memsize) // then free it

					src.bind(dv)
					// the CPU method is correct. This method is correct for MOST cases, but will not be correct under some other circumstances
					// ctx.MemcpyDtoH(dv.d.Pointer(), cu.DevicePtr(v.Uintptr()), instr.size)
				}
			}
		}

	}

	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v.Uintptr())
	m.leaveLogScope()

	return nil
}

func (instr deviceTransport) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v", instr)
	from := m.getValue(instr.from)
	to := m.getValue(instr.to)

	var ctx *cu.BatchedContext
	switch {
	case instr.from.device == CPU && instr.to.device != CPU:
		memsize := int64(from.MemSize())
		ctx = m.Contexts()[int(instr.to.device)]
		ctx.MemcpyHtoD(cu.DevicePtr(to.Uintptr()), from.Pointer(), memsize)
	case instr.from.device != CPU && instr.to.device == CPU:
		dt := from.Dtype()
		memsize := calcMemSize(dt, from.Shape())
		ctx = m.Contexts()[int(instr.from.device)]
		ctx.MemcpyDtoH(to.Pointer(), cu.DevicePtr(from.Uintptr()), memsize)

		// when copying from device to host, it's assumed that the host will want to immediately use
		// so signal the DoWork
		m.Signal()
	}

	return nil
}
