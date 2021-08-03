package cuda

import (
	"fmt"
	"log"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
	cudalib "gorgonia.org/gorgonia/cuda"
)

// this file relates to code that allows you to extend Engine

func (e *Engine) loadStdLib() error {
	stdlib := cudalib.StandardLib()

	for _, l := range stdlib {
		log.Printf("Loading %v", l.ModuleName)
		if err := e.LoadCUDAFunc(l.ModuleName, l.Data, l.Funcs); err != nil {
			return err
		}
	}
	return nil
}

// LoadCUDAFunc loads a string representing a CUDA PTX file into the engine, giving it the universe of computing functions.
func (e *Engine) LoadCUDAFunc(moduleName, data string, funcs []string) (err error) {
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
