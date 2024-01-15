package cuda

import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	cudalib "gorgonia.org/gorgonia/cuda"
	"gorgonia.org/internal/debug"
)

// this file relates to code that allows you to extend Engine

// LoadStdLib loads the standard lib from the gorgonia standard library.
// This function needs to be called in the same threadlocked OS thread as the thread
// that created the engine.
func LoadStdLib(e *EngineState) error {
	stdlib := cudalib.StandardLib()
	debug.Logtid("LoadStdLib", 0)
	debug.Logf("Length of stdlib %d", len(stdlib))

	for _, l := range stdlib {
		if err := e.LoadCUDAFunc(l.ModuleName, l.Data, l.Funcs); err != nil {
			return err
		}
	}
	return nil
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the engine, giving it the universe of computing functions.
func (e *EngineState) LoadCUDAFunc(moduleName, data string, funcs []string) (err error) {
	fns := e.f
	if fns == nil {
		fns = make(map[string]cu.Function)
	}
	if err = cu.SetCurrentContext(e.c.Context.CUDAContext()); err != nil {
		return errors.Wrapf(err, "Unable to set current context when loading module %q at device %v", moduleName, e.d)
	}

	var mod cu.Module
	if mod, err = cu.LoadData(data); err != nil {
		return errors.Wrapf(err, "Failed to load module %q data for Device %v context %x", moduleName, e.d, e.c)
	}

	for _, name := range funcs {
		var fn cu.Function
		if fn, err = mod.Function(name); err != nil {
			return errors.Wrapf(err, "Unable to get function %q in Device %v context %x", name, e.d, e.c)
		}
		fqn := fmt.Sprintf("%v.%v", moduleName, name)
		fns[fqn] = fn
	}
	if e.m == nil {
		e.m = make(map[string]cu.Module)
	}
	e.m[moduleName] = mod
	e.f = fns
	return nil
}

// Call launches a known kernel that takes at least one argument.
// The argument's size must be known.
func (e *EngineState) Call(fnName string, size int, args ...unsafe.Pointer) error {
	debug.Logf("e.Call %v", fnName)
	if !e.HasFunc(fnName) {
		return errors.Errorf("The engine does not have the function %q", fnName)
	}

	fn := e.f[fnName]
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(size)

	debug.Logf("gx %d, gy %d, gz %d | bx %d by %d, bz %d", gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ)
	debug.Logf("args: %v", args)
	debug.Logf("Call %v", fnName)
	debug.Logtid("e.Call", 1)
	e.c.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	return nil
}
