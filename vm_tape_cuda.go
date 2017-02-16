// +build cuda

package gorgonia

import (
	"log"

	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

type modules map[string][]cu.Module // ths size of the slice has to be the same as the slice of contexts
type contexts []cu.Context

func (m modules) HasFunc(name string) bool {
	_, ok := m[name]
	return ok
}

func (m modules) Function(name string) (interface{}, error) {
	mod, ok := m[name]
	if !ok {
		return nil, errors.Errorf("Function %q not found", name)
	}
	return mod, nil
}

func finalizeTapeMachine(m *tapeMachine) {
	for i, c := range m.c {
		cu.SetCurrent(c)
		for _, v := range m.m {
			mod := v[i]
			cu.Unload(mod)
		}
		cu.DestroyContext(&c)
	}
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
	if !initCUDA {
		// don't bother initializing contexts if no instructions were CUDA based
		return
	}

	devices, _ := cu.NumDevices()
	m.c = make(contexts, devices)
	for i := range m.c {
		dev, _ := cu.GetDevice(i)
		ctx, _ := dev.MakeContext(cu.SchedAuto)
		m.c[i] = ctx
	}
	if len(m.c) > 0 {
		cu.SetCurrent(m.c[0])
	}

	m.m = make(modules)
	m.loadStdLib()
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the machine.
//
// The convention is to have one function per module, sharing the same name.
func (m *tapeMachine) LoadCUDAFunc(name, data string) (err error) {
	if len(m.c) == 0 {
		return nil
	}

	mods := make([]cu.Module, len(m.c))
	for i, c := range m.c {
		if err = cu.SetCurrent(c); err != nil {
			err = errors.Wrapf(err, "Unable to set current context when loading module %q at context %d", name, i)
			return
		}
		var mod cu.Module
		if mod, err = cu.LoadData(data); err != nil {
			err = errors.Wrapf(err, "Failed to load module %q data for context %d", name, i)
			return
		}
		mods[i] = mod
	}
	// set the first to current
	if len(m.c) > 0 {
		cu.SetCurrent(m.c[0])
	}
	m.m[name] = mods
	return nil
}

func (m *tapeMachine) Contexts() []cu.Context {
	return []cu.Context(m.c)
}

func (m *tapeMachine) Modules() map[string][]cu.Module {
	return map[string][]cu.Module(m.m)
}

// loads the standardlib
func (m *tapeMachine) loadStdLib() {
	if cudaStdLib == nil {
		return
	}

	for name, data := range cudaStdLib {
		if err := m.LoadCUDAFunc(name, data); err != nil {
			log.Printf("err %v", err)
		}
	}
}
