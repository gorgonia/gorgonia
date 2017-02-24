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
	m.cleanup()
}

func (m *tapeMachine) init() {
	var initCUDA bool
	for _, instr := range m.p.instructions {
		if eo, ok := instr.(execOp); ok {
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
		log.Printf("m.c %v", m.c)
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
