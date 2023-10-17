package values

import (
	"gorgonia.org/dtype"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/tensor"
)

func tensorInfo(t tensor.Desc) (dt dtype.Dtype, dim int) {
	dt = t.Dtype()
	dim = t.Dims()
	return
}

func makeScalarFromMem(dt dtype.Dtype, mem tensor.Memory) (retVal Value, err error) {
	switch dt {
	/*
		case tensor.Float64:
			retVal = (*F64)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Float32:
			retVal = (*F32)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Int:
			retVal = (*I)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Int64:
			retVal = (*I64)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Int32:
			retVal = (*I32)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Byte:
			retVal = (*U8)(unsafe.Pointer(mem.Uintptr()))
		case tensor.Bool:
			retVal = (*B)(unsafe.Pointer(mem.Uintptr()))
	*/
	default:
		err = gerrors.NYI(dt)
		//err = errors.Errorf(gerrors.NYITypeFail, "makeScalarFromMem", dt)
	}
	return
}
