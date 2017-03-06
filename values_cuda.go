// +build cuda

package gorgonia

import (
	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

func valToDevicePointer(ctx *cu.BatchedContext, val Value) (mem cu.DevicePtr, err error) {
	// alloc:
	size := int64(val.MemSize())
	if mem, err = cu.MemAlloc(size); err != nil {
		err = errors.Wrapf(err, "Cannot get mem device pointer")
		return

	}

	// batched copy
	if ctx != nil {
		ctx.MemcpyHtoD(mem, val.Pointer(), size)
		return
	}

	// blocking copy
	if err = cu.MemcpyHtoD(mem, val.Pointer(), size); err != nil {
		err = errors.Wrapf(err, "Memcpy failed")
		return
	}
	return mem, nil
}

func devPtrToValue(ctx *cu.BatchedContext, val Value, mem cu.DevicePtr) (err error) {
	size := int64(val.MemSize())
	ptr := val.Pointer()
	if ctx != nil {
		ctx.MemcpyDtoH(ptr, mem, size)
		ctx.DoWork()
		return nil
	}
	return cu.MemcpyDtoH(ptr, mem, size)
}
