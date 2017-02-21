// +build cuda

package gorgonia

import (
	"unsafe"

	"github.com/chewxy/cu"
	"github.com/chewxy/gorgonia/tensor"
	"github.com/pkg/errors"
)

func valToDevicePointer(val Value) (mem cu.DevicePtr, err error) {
	switch v := val.(type) {
	case *tensor.Dense:
		size := int64(v.DataSize() * int(v.Dtype().Size()))
		if mem, err = cu.MemAlloc(size); err != nil {
			err = errors.Wrapf(err, "Cannot get mem device pointer")
			return
		}
		switch v.Dtype() {
		case tensor.Float64:
			data := v.Data().([]float64)
			if err = cu.MemcpyHtoD(mem, unsafe.Pointer(&data[0]), size); err != nil {
				err = errors.Wrapf(err, "Memcpy failed")
				return
			}
			return
		case tensor.Float32:
			data := v.Data().([]float32)
			if err = cu.MemcpyHtoD(mem, unsafe.Pointer(&data[0]), size); err != nil {
				err = errors.Wrapf(err, "Memcpy failed")
				return
			}
			return
		default:
			if err = cu.MemFree(mem); err != nil {
				err = errors.Wrapf(err, "Unable to free mem properly")
				return
			}
			return 0, errors.Errorf(unsupportedDtype, v.Dtype())
		}

	default:
	}
	return 0, errors.Errorf("Cannot convert %T to device pointer", val)
}

func devPtrToValue(val Value, mem cu.DevicePtr) (err error) {
	switch v := val.(type) {
	case *tensor.Dense:
		size := int64(v.DataSize() * int(v.Dtype().Size()))
		switch v.Dtype() {
		case tensor.Float64:
			data := v.Data().([]float64)
			if err = cu.MemcpyDtoH(unsafe.Pointer(&data[0]), mem, size); err != nil {
				err = errors.Wrapf(err, "Memcpy failed")
				return
			}
			return nil
		case tensor.Float32:
			data := v.Data().([]float32)
			if err = cu.MemcpyDtoH(unsafe.Pointer(&data[0]), mem, size); err != nil {
				err = errors.Wrapf(err, "Memcpy failed")
				return
			}
			return nil
		default:
			return errors.Errorf(unsupportedDtype, v.Dtype())
		}
	default:
	}
	return errors.Errorf("Cannot copy memory from device to %T", val)
}
