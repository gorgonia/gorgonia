// +build cuda

package gorgonia

import (
	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

// Device represents the device where the code will be executed on. It can either be a GPU or CPU
type Device cu.Device

// CPU is the default the graph will be executed on.
const CPU = Device(cu.CPU)

// String implements fmt.Stringer and runtime.Stringer
func (d Device) String() string { return cu.Device(d).String() }

// Alloc allocates memory on the device. If the device is CPU, the allocations is a NO-OP because Go handles all the allocations in the CPU
func (d Device) Alloc(extern External, size int64) (tensor.Memory, error) {
	if d == CPU {
		cudaLogf("device is CPU")
		return nil, nil // well there should be an error because this wouldn't be called
	}

	machine := extern.(CUDAMachine)
	ctxes := machine.Contexts()
	if len(ctxes) == 0 {
		cudaLogf("allocate nothing")
		return nil, nil
	}
	ctx := ctxes[int(d)]

	cudaLogf("calling ctx.MemAlloc(%d)", size)
	return ctx.MemAlloc(size)
}

// Free the memory of the device
func (d Device) Free(extern External, mem tensor.Memory, size int64) (err error) {
	var devptr cu.DevicePtr
	var ok bool
	if devptr, ok = mem.(cu.DevicePtr); !ok {
		return nil
	}

	machine := extern.(CUDAMachine)
	machine.Put(d, devptr, size)

	// FUTURE: actually free memory if there ain't enough to go round

	// ctx := machine.Contexts()[int(d)]
	// cudaLogf("MemFree %v", devptr)
	// ctx.MemFree(devptr)
	return nil
}
