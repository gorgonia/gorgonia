// +build !cuda

package gorgonia

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

func finalizeTapeMachine(m *tapeMachine) {}

// UseCudaFor is an option for *tapeMachine. This function is NO-OP unless the program is built with the `cuda` tag.
func UseCudaFor(ops ...string) VMOpt {
	return func(m VM) {}
}

func (m *tapeMachine) getEngine(dev Device) tensor.Engine { return m.Engine }

func (instr *execOp) exec(m *tapeMachine) (err error) {
	m.logf("Executing %v. Node is: %x", instr, instr.id)
	m.enterLogScope()
	defer m.leaveLogScope()

	// Read
	m.watchedLogf("Inputs:")
	m.enterLogScope()
	var inputs []Value
	for _, reg := range instr.readFrom {
		v := m.cpumem[reg.id]
		inputs = append(inputs, v)
		m.watchedLogf(m.valueFmt, v)
	}
	m.leaveLogScope()

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

	m.watchedLogf("Result:")
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()
	// TODO: type and shape checks

	// Write
	setEngine(v, m.Engine)
	dest := instr.writeTo.id
	m.cpumem[dest] = v
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
	m.enterLogScope()
	m.watchedLogf(m.valueFmt, v)
	m.leaveLogScope()
	return nil
}

func (instr deviceTransport) exec(m *tapeMachine) error {
	return nil
}
