// +build cuda

package gorgonia

import (
	"log"

	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

// UseCudaFor is an option for *tapeMachine only. At the moment users should pass in strings of the op name ("add", "sub"...)
// Do not pass in the types (for example, don't pass in "add64")
func UseCudaFor(ops ...string) VMOpt {
	f := func(m VM) {
		switch v := m.(type) {
		case *tapeMachine:
			if v.c == nil {
				v.init()
			}

			if len(ops) == 0 {
				v.loadStdLib()
				return
			}

			for _, op := range ops {
				op64 := op + "64"
				op32 := op + "32"

				cudaLogf("Trying to load %q and %q. m.c = %v", op64, op32, v.c)

				if data, ok := cudaStdLib[op64]; ok {
					if err := v.LoadCUDAFunc(op64, data); err != nil {
						log.Printf("Unable to load %q: %v", op64, err)
					}
				}

				if data, ok := cudaStdLib[op32]; ok {
					if err := v.LoadCUDAFunc(op32, data); err != nil {
						log.Printf("Unable to load %q: %v", op32, err)
					}
				}
			}
		}
	}
	return f
}

func finalizeTapeMachine(m *tapeMachine) {
	cudaLogf("Finalizing tape machine %p", m)
	for i, c := range m.c {
		cu.SetCurrent(c.Context)
		for _, v := range m.m {
			mod := v[i]
			cu.Unload(mod)
		}
		cu.DestroyContext(&c.Context)
	}
	m.Cleanup()
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
	m.ExternMetadata.init(m.p.gpumem)
	cudaLogf("m.c = %v", m.c)
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the machine.
//
// The convention is to have one function per module, sharing the same name.
func (m *tapeMachine) LoadCUDAFunc(name, data string) (err error) {
	if len(m.c) == 0 {
		return nil
	}

	mods := make([]cu.Module, len(m.c))
	fns := make([]cu.Function, len(m.c))
	for i, c := range m.c {
		if err = cu.SetCurrent(c.Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current context when loading module %q at context %d", name, i)
			return
		}

		var mod cu.Module
		if mod, err = cu.LoadData(data); err != nil {
			err = errors.Wrapf(err, "Failed to load module %q data for %dth context %x", name, i, c)
			return
		}

		var fn cu.Function
		if fn, err = mod.Function(name); err != nil {
			err = errors.Wrapf(err, "Unable to get function %q in %dth context %x", name, i, c)
			return
		}
		mods[i] = mod
		fns[i] = fn
	}

	// set the first to current
	if len(m.c) > 0 {
		if err = cu.SetCurrent(m.c[0].Context); err != nil {
			err = errors.Wrapf(err, "Unable to set current")
			return
		}
	}

	m.m[name] = mods
	m.f[name] = fns
	cudaLogf("Loaded %q", name)
	return nil
}

// loads the standardlib
func (m *tapeMachine) loadStdLib() {
	if cudaStdLib == nil {
		return
	}

	for name, data := range cudaStdLib {
		if err := m.LoadCUDAFunc(name, data); err != nil {
			cudaLogf("Unable to load %q.: %v", name, err)
		}
	}
}

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLoggingContext()
	defer m.leaveLoggingContext()

	enterLoggingContext()
	defer leaveLoggingContext()

	m.watchedLogf("Inputs:")
	m.enterLoggingContext()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.getValue(reg)
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLoggingContext()

	toDev := instr.writeTo.device
	var v Value
	switch op := instr.op.(type) {
	case CUDADoer:
		prealloc := m.getValue(instr.writeTo)
		if v, err = op.CUDADo(m, toDev, prealloc, inputs...); err != nil {
			return errors.Wrapf(err, "Happened while attempting to use CUDA to execute %v. Node is %x. Register was %v", instr, instr.id, instr.writeTo.id)
		}
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

	}
	m.watchedLogf("Result:")
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()
	// TODO: type and shape checks

	// Write
	m.writeValue(instr.writeTo, v)
	node := m.p.g.Node(instr.id).(*Node)

	if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
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

				add := newEBOByType(addOpType, TypeOf(dv.d), TypeOf(v))

				if d, err := add.UnsafeDo(dv.d, v); err == nil {
					dv.SetDeriv(d)
					src.bind(dv)
				} else {
					return err
				}
			}
		}

	}

	m.watchedLogf("Written To: %v", instr.writeTo)
	m.enterLoggingContext()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLoggingContext()

	return nil
}

// func (instr *readInstr) exec(m *tapeMachine) (err error) {
// 	m.logf("Executing READ - read from %v", instr.readFrom)
// 	v := m.getValue(instr.readFrom)
// 	if v == nil {
// 		m.logf("ERR1")
// 		return errors.Errorf(nyiFail, "Cannot read instruction")
// 	}

// 	var v2 Value
// 	if instr.readFrom.device != CPU && !instr.s.IsScalar() {
// 		var dt tensor.Dtype
// 		if dt, err = dtypeOf(instr.t); err != nil {
// 			return errors.Wrapf(err, dtypeExtractionFail, instr.t)
// 		}
// 		vt := tensor.New(tensor.Of(dt), tensor.WithShape(instr.s...))

// 		// ctx := m.Contexts()[int(instr.readFrom.device)]
// 		// ctx.MemcpyDtoH(vt.Pointer(), cu.DevicePtr(v.Uintptr()), int64(vt.MemSize()))

// 		v2 = vt

// 	} else {
// 		if v2, err = CloneValue(v); err != nil {
// 			m.logf("ERR2")
// 			return errors.Wrap(err, cloneFail)
// 		}

// 	}

// 	m.logf("v2 : %v %v", v2, v2.Uintptr())

// 	*instr.into = v2
// 	return nil
// }

func (instr deviceTransport) exec(m *tapeMachine) (err error) {
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
		memsize := int64(from.Shape().TotalSize()) * int64(dt.Size())
		ctx = m.Contexts()[int(instr.from.device)]
		ctx.MemcpyDtoH(to.Pointer(), cu.DevicePtr(from.Uintptr()), memsize)
	}

	return nil
}
