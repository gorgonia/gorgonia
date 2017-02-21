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
		cu.SetCurrent(c)
		for _, v := range m.m {
			mod := v[i]
			cu.Unload(mod)
		}
		cu.DestroyContext(&c)
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
		return
	}

	devices, _ := cu.NumDevices()
	m.c = make([]cu.Context, devices)
	m.d = make([]cu.Device, devices)
	m.warp = make([]int, devices)
	m.mtpb = make([]int, devices)
	m.mgdx = make([]int, devices)
	m.mgdy = make([]int, devices)
	m.mgdz = make([]int, devices)
	m.mbdx = make([]int, devices)
	m.mbdy = make([]int, devices)
	m.mbdz = make([]int, devices)
	for i := range m.c {
		dev, err := cu.GetDevice(i)
		if err != nil {
			cudaLogf("Failed to get device %d: %v", i, err)
			m.cleanup()
			return
		}
		ctx, err := dev.MakeContext(cu.SchedAuto)
		if err != nil {
			if err == cu.OutOfMemory {
				var free, total int64
				if free, total, err = cu.MemInfo(); err != nil {
					cudaLogf("Error while getting mem info: %v", err)
				}
				cudaLogf("Out of memory. Free: %v, total %v", free, total)
				m.cleanup()
				return
			}
			cudaLogf("Failed to make context for device %d. Error: %v", i, err)
			m.cleanup()
			return
		}

		var attrs []int
		if attrs, err = dev.Attributes(cu.WarpSize, cu.MaxThreadsPerBlock, cu.MaxGridDimX, cu.MaxGridDimY, cu.MaxGridDimZ, cu.MaxBlockDimX, cu.MaxBlockDimY, cu.MaxBlockDimZ); err != nil {
			cudaLogf("Failed to get attributes for device %d. Error: %v", i, err)
			m.cleanup()
			return
		}

		m.warp[i] = attrs[0]
		m.mtpb[i] = attrs[1]
		m.mgdx[i] = attrs[2]
		m.mgdy[i] = attrs[3]
		m.mgdz[i] = attrs[4]
		m.mbdx[i] = attrs[5]
		m.mbdy[i] = attrs[6]
		m.mbdz[i] = attrs[7]

		m.c[i] = ctx
		m.d[i] = dev
	}
	if len(m.c) > 0 {
		cu.SetCurrent(m.c[0])
	}
	m.m = make(map[string][]cu.Module)
	m.f = make(map[string][]cu.Function)
	// m.loadStdLib()

	// var free, total int64
	// var err error
	// if free, total, err = cu.MemInfo(); err != nil {
	// 	cudaLogf("Error while getting mem info: %v", err)
	// }
	// cudaLogf("Machine %p initialized. CUDA Memory: %v/%v", m, free, total)
}

func (m *tapeMachine) cleanup() {
	m.c = nil
	m.m = nil
	m.f = nil
	m.d = nil
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
		if err = cu.SetCurrent(c); err != nil {
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
		if err = cu.SetCurrent(m.c[0]); err != nil {
			err = errors.Wrapf(err, "Unable to set current")
			return
		}
	}

	m.m[name] = mods
	m.f[name] = fns
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
