package cuda

import "C"

import (
	"fmt"
	"unsafe"

	"github.com/pkg/errors"
	"gorgonia.org/cu"
)

// this file provides checker methods for the engine.

// NaNChecker checks that the tensor contains a NaN
func (e *Engine[DT, T]) HasNaN(a T) (bool, error) {
	dt := a.Dtype()
	name := fmt.Sprintf("misc.hasNaN_f%v", int(dt.Size()*8))

	if !e.HasFunc(name) {
		return false, errors.Errorf("Unable to perform HasNaN(). The tensor engine does not have the function %q", name)
	}

	mem := cu.DevicePtr(a.Uintptr())
	size := int64(logicalSize(a.Shape()))
	fn := e.f[name]

	var retVal C.int
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
		unsafe.Pointer(&retVal),
	}
	e.c.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	e.Signal()
	return int(retVal) > 0, e.c.Error()
}

// InfChecker checks that the tensor contains a Inf
func (e *Engine[DT, T]) HasInf(a T) (bool, error) {
	dt := a.Dtype()
	name := fmt.Sprintf("misc.hasInf_f%v", int(dt.Size()*8))

	if !e.HasFunc(name) {
		return false, errors.Errorf("Unable to perform HasInf(). The tensor engine does not have the function %q", name)
	}

	mem := cu.DevicePtr(a.Uintptr())
	size := int64(logicalSize(a.Shape()))
	fn := e.f[name]

	var retVal C.int
	gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ := e.ElemGridSize(int(size))
	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
		unsafe.Pointer(&retVal),
	}
	e.c.LaunchAndSync(fn, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, cu.NoStream, args)
	e.Signal()
	return int(retVal) > 0, e.c.Error()
}
