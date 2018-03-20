// +build !cuda

package gorgonia

import "github.com/pkg/errors"

func finalizeTapeMachine(m *tapeMachine) {}

// UseCudaFor is an option for *tapeMachine. This function is NO-OP unless the program is built with the `cuda` tag.
func UseCudaFor(ops ...string) VMOpt {
	return func(m VM) {}
}

func (instr *execOp) execKernel(m *tapeMachine, inputs []Value) (err error) {
	pc := int(instr.ID())
	// Execute
	var v Value
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

	m.watchedPCLogf(pc, "Result:")
	m.enterLogScope()
	m.watchedPCLogf(pc, m.valueFmt, v)
	m.leaveLogScope()
	// TODO: type and shape checks

	// Write
	setEngine(v, m.Engine)
	m.writeValue(instr.writeTo, v)

	// additional processing
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

	m.watchedPCLogf(pc, "Written To: %v", instr.writeTo)
	m.enterLogScope()
	m.watchedPCLogf(pc, m.valueFmt, v)
	m.leaveLogScope()
	return nil
}

func (instr deviceTransport) exec(m *tapeMachine) error {
	return nil
}
