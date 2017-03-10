// +build cuda

package gorgonia

import (
	"log"

	"github.com/chewxy/cu"
	"github.com/chewxy/gorgonia/tensor"
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
	m.cleanup()
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
	m.ExternMetadata.init()
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

	if instr.useGPU {
		if len(m.Contexts()) == 0 {
			goto usecpu
		}

		// Read
		m.watchedLogf("Inputs:")
		m.enterLoggingContext()
		inputs := make([]Memory, len(instr.readFrom))
		fromDevs := make([]Device, len(instr.readFrom))
		for i, reg := range instr.readFrom {
			inputs[i] = m.getMemory(reg)
			fromDevs[i] = reg.device
			cudaLogf("inputs[%d] :%T", i, inputs[i])
			m.watchedLogf(m.valueFmt, inputs[i])
		}
		m.leaveLoggingContext()

		toDev := instr.writeTo.device

		// Execute
		var mem Memory
		switch cd := instr.op.(type) {
		case CUDADoer:
			prealloc := m.getMemory(instr.writeTo)
			if mem, err = cd.CUDADo(m, toDev, instr.ExecutionMetadata, prealloc, inputs...); err != nil {
				return errors.Wrapf(err, "Happened while attempting to use CUDA to execute %v. Node is %x. Register was %v", instr, instr.id, instr.writeTo.id)
			}

			cudaLogf("prealloc mem: %v", mem)
		case CLDoer:
			goto usecpu
		default:
			goto usecpu
		}

		// Write
		var v Value
		var convertedFromMem bool
		dest := instr.writeTo.id
		node := m.p.g.Node(instr.id).(*Node)
		ctx := m.c[int(toDev)]
		switch mt := mem.(type) {
		case Value:
			v = mt
		case cu.DevicePtr:
			v = node.Value()
			if v == nil {
				cudaLogf("allocating v")
				// create v
				switch t := instr.OutputType.(type) {
				case TensorType:
					var dt tensor.Dtype
					if dt, err = dtypeOf(t); err != nil {
						err = errors.Wrapf(err, "execOp cannot get dtype out of %v", t)
						return // very unlikely to happen
					}

					v = tensor.New(tensor.Of(dt), tensor.WithShape(instr.OutputShape...))
					if err = devPtrToValue(ctx, v, mt); err != nil {
						err = errors.Wrap(err, "execOp cannot copy mem to value")
						return
					}
				case Scalar:
					// wtf?
				}
				cudaLogf("Done allocating v")
			} else {
				cudaLogf("copying v")
				// copy
				if err = devPtrToValue(ctx, v, mt); err != nil {
					return
				}
			}
			convertedFromMem = true
		}

		switch instr.writeTo.device {
		case CPU:
			cudaLogf("write to cpu register")
			m.cpumem[dest] = v
		default:
			cudaLogf("write %v to GPU register", mem)
			m.gpumem[dest] = mem
		}

		if m.trace() && (len(m.watchNodes) == 0 || m.watchNodes.Contains(node)) {
			if err = node.bindCopy(v); err != nil {
				return errors.Wrapf(err, "TraceExec failed to bind copy")
			}
		} else {
			cudaLogf("bind v to node %v", node.Name())
			cudaLogf("v %p %v", v, v.Pointer())
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
		m.watchedLogf("Written To: %v | Converted from Memory %t | v: %v", instr.writeTo, convertedFromMem, v.Pointer())
		m.enterLoggingContext()
		m.watchedLogf(m.valueFmt, v)
		m.leaveLoggingContext()
		return nil
	}

usecpu:
	cudaLogf("Using CPU for %v", instr.op)

	// Read
	m.watchedLogf("Inputs:")
	m.enterLoggingContext()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v, mem := m.getValue(reg)
		switch {
		case v == nil && mem != nil:
			dev := reg.device
			ctx := m.Contexts()[int(dev)]

			// walk the instructions backwards
			var prev tapeInstr
			for i := m.pc - 1; i >= 0; i-- {
				prev = m.p.instructions[i]
				if prev.writes() == reg {
					break
				}
			}
			var n *Node
			var f fragment
			for n, f = range m.p.m {
				if f.has(prev) {
					break
				}
			}
			v := n.Value()
			devPtrToValue(ctx, v, mem.(cu.DevicePtr))

			cudaLogf("using n.Value: \n%v", n.Value())
			inputs = append(inputs, n.Value())
			continue

		case v != nil:
			inputs = append(inputs, v)
		case v == nil && m == nil:
			err = errors.Errorf("Cannot extract from nil memory")
			return
		}

		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLoggingContext()

	// Execute
	var v Value
	switch {
	case instr.preAllocated:
		cudaLogf("preallocated. instr.writeTo %v", instr.writeTo)
		if pd, ok := instr.op.(UsePreallocDoer); ok {
			p, _ := m.getValue(instr.writeTo)
			if p == nil {
				if v, err = instr.op.Do(inputs...); err != nil {
					return errors.Wrapf(err, opDoFail)
				}
			} else {
				cudaLogf("WTF??! %v %v", p, p == nil)

				if v, err = pd.UsePreallocDo(p, inputs...); err != nil {
					return errors.Wrapf(err, "Happened while attempting to execute %v. Node is %x. Register was: %v ", instr, instr.id, instr.writeTo)
				}
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
