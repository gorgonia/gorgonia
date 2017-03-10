// +build cuda

package gorgonia

import (
	"github.com/chewxy/cu"
	"github.com/pkg/errors"
)

//  convM2V converts Memory to Value
func convM2V(m External, dev Device, mem Memory, val *Value) (err error) {
	switch mt := mem.(type) {
	case Value:
		*val = mt
	case cu.DevicePtr:
		machine := m.(CUDAMachine)
		ctxes := machine.Contexts()
		if len(ctxes) == 0 || len(ctxes) <= int(dev) {
			return errors.Errorf("Cannot convert Memory to Value when there are no CUDA contexts")
		}
		ctx := ctxes[int(dev)]
		if err = devPtrToValue(ctx, *val, mt); err != nil {
			return
		}
	}
	return nil
}

func valToDevicePointer(ctx *cu.BatchedContext, val Value) (mem cu.DevicePtr, err error) {
	// alloc:
	size := int64(val.MemSize())
	if ctx != nil {
		return ctx.AllocAndCopy(val.Pointer(), size)
	}

	// otherwise do blocking copy
	if mem, err = cu.MemAlloc(size); err != nil {
		err = errors.Wrapf(err, "Cannot get mem device pointer")
		return
	}

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
		// ctx.DoWork()
		// runtime.Gosched()
		return nil
	}
	return cu.MemcpyDtoH(ptr, mem, size)
}
