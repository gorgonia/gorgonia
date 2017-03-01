// +build cuda

package gorgonia

import "github.com/chewxy/cu"

// Device represents the device where the code will be executed on. It can either be a GPU or CPU
type Device cu.Device

// CPU is the default the graph will be executed on.
const CPU = Device(cu.CPU)

// String implements fmt.Stringer and runtime.Stringer
func (d Device) String() string { return cu.Device(d).String() }

// Alloc allocates memory on the device. If the device is CPU, the allocations is a NO-OP because Go handles all the allocations in the CPU
func (d Device) Alloc(extern External, size int64) (Memory, error) {
	if d == CPU {
		return nil, nil // well there should be an error because this wouldn't be called
	}

	machine := extern.(CUDAMachine)
	ctx := machine.Contexts()[int(d)]
	// TODO in the future push and pop contexts instead
	if err := cu.SetCurrent(ctx.Context); err != nil {
		return nil, err
	}
	return cu.MemAlloc(size)
}

func (d Device) Free(extern External, mem Memory) (err error) {
	var devptr cu.DevicePtr
	var ok bool
	if devptr, ok = mem.(cu.DevicePtr); !ok {
		return nil
	}

	machine := extern.(CUDAMachine)
	ctx := machine.Contexts()[int(d)]

	ctx.MemFree(devptr)
	return nil
}
