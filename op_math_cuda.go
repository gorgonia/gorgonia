// +build cuda

package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

func (op elemUnaryOp) CallsExtern() bool { return true }

func (op elemUnaryOp) CUDADo(extern External, fromDevs []Device, toDev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	a := inputs[0]
	if a.Shape().IsScalar() {
		return op.do(inputs)
	}

	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(DtypeOf(a).Size())*8)
	if !extern.HasFunc(name) {
		return op.Do(inputs...)
	}

	// TODO: maybe check arity of fromDevs?
	dev := toDev

	machine := extern.(CUDAMachine)
	ctxes := machine.Contexts()
	mods := machine.Modules()
	if len(ctxes) == 0 {
		return op.Do(inputs...) // resort to using slow methods if no devices were found
	}
	mod := mods[name][int(dev)]
	var fn cu.Function
	if fn, err = mod.Function(name); err != nil {
		return op.Do(inputs...)
	}

	size := a.Shape().TotalSize()

	if retVal, err = Copy(prealloc, a); err != nil {
		return op.Do(inputs...)
	}

	// TODO: optimize kernel to maximize parallelization
	// var maxThreads int
	// d := cu.Device(dev)
	// if maxThreads, err = d.Attribute(cu.MaxThreadsPerBlock); err != nil {
	// 	return op.Do(inputs...) // resort to using slow methods if there was an error
	// }

	var mem cu.DevicePtr
	if mem, err = valToDevicePointer(retVal); err != nil {
		err = errors.Wrapf(err, opDoFail)
		return
	}
	defer cu.MemFree(mem) // no leaks plz

	args := []unsafe.Pointer{
		unsafe.Pointer(&mem),
		unsafe.Pointer(&size),
	}

	if err = fn.LaunchAndSync(1, 1, 1, size, 1, 1, 0, cu.Stream(0), args); err != nil {
		return
	}

	err = devPtrToValue(retVal, mem)
	return
}

func (op elemUnaryOp) CUDAFuncName() string {
	return op.String()
}

func (op elemBinOp) CallsExtern() bool { return true }

func (op elemBinOp) CUDADo(extern External, fromDevs []Device, toDev Device, prealloc Value, inputs ...Value) (retVal Value, err error) {
	if err = checkArity(op, len(inputs)); err != nil {
		return
	}

	if !op.isArith() {
		if prealloc != nil && !prealloc.Shape().IsScalar() {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		return op.Do(inputs...)
	}

	a := inputs[0]
	b := inputs[1]
	if a.Shape().IsScalar() || b.Shape().IsScalar() || prealloc == nil || prealloc.Shape().IsScalar() {
		return op.Do(inputs...)
	}

	name := fmt.Sprintf("%v%d", op.CUDAFuncName(), int(DtypeOf(a).Size())*8)
	if !extern.HasFunc(name) {
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		return op.Do(inputs...)
	}

	// TODO: maybe check arity of fromDevs?
	dev := toDev

	machine := extern.(CUDAMachine)
	ctxes := machine.Contexts()
	mods := machine.Modules()
	if len(ctxes) == 0 {
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		return op.Do(inputs...)
	}
	mod := mods[name][int(dev)]
	var fn cu.Function
	if fn, err = mod.Function(name); err != nil {
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		return op.Do(inputs...)
	}

	size := a.Shape().TotalSize()

	if retVal, err = Copy(prealloc, a); err != nil {
		// TODO: maybe warn?
		if prealloc != nil {
			return op.UsePreallocDo(prealloc, inputs...)
		}
		return op.Do(inputs...)
	}

	// TODO: optimize kernel to maximize parallelization
	// var maxThreads int
	// d := cu.Device(dev)
	// if maxThreads, err = d.Attribute(cu.MaxThreadsPerBlock); err != nil {
	// 	return op.Do(inputs...) // resort to using slow methods if there was an error
	// }

	var memA, memB cu.DevicePtr
	if memA, err = valToDevicePointer(retVal); err != nil {
		err = errors.Wrapf(err, opDoFail)
		return
	}

	if memB, err = valToDevicePointer(b); err != nil {
		err = errors.Wrapf(err, opDoFail)
		return
	}

	// we don't want no leaks
	defer cu.MemFree(memB)
	defer cu.MemFree(memA)

	args := []unsafe.Pointer{
		unsafe.Pointer(&memA),
		unsafe.Pointer(&memB),
		unsafe.Pointer(&size),
	}

	if err = fn.LaunchAndSync(1, 1, 1, size, 1, 1, 0, cu.Stream(0), args); err != nil {
		return
	}

	err = devPtrToValue(retVal, memA)
	return

}

func (op elemBinOp) CUDAFuncName() string {
	return ʘBinOpNames[op.binOpType()]
}
