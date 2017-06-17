package tensor

import (
	"reflect"
	"unsafe"
)

// ConsOpt is a tensor construction option.
type ConsOpt func(Tensor)

// Of is a construction option for a Tensor.
func Of(a Dtype) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			tt.t = a
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithBacking is a construction option for a Tensor
// Use it as such:
//		backing := []float64{1,2,3,4}
// 		t := New(WithBacking(backing))
// It can be used with other construction options like WithShape
func WithBacking(x interface{}, argMask ...[]bool) ConsOpt {
	var mask []bool
	if len(argMask) > 0 {
		mask = argMask[0]
	}
	f := func(t Tensor) {
		if x == nil {
			return
		}
		switch tt := t.(type) {
		case *Dense:
			tt.fromSlice(x)
			if len(argMask) > 0 {
				tt.addMask(argMask[0])
			}
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithMask is a construction option for a Tensor
// Use it as such:
//		mask := []bool{true,true,false,false}
// 		t := New(WithBacking(backing))
// It can be used with other construction options like WithShape
// The supplied mask can be any type. If non-boolean, then tensor mask is set to true
// wherever non-zero value is obtained
func WithMask(x interface{}) ConsOpt {
	f := func(t Tensor) {
		if x == nil {
			return
		}
		switch tt := t.(type) {
		case *Dense:
			tt.MaskFromSlice(x)
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithShape is a construction option for a Tensor. It creates the ndarray in the required shape.
func WithShape(dims ...int) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			throw := BorrowInts(len(dims))
			copy(throw, dims)
			tt.setShape(throw...)
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// FromScalar is a construction option for representing a scalar value as a Tensor
func FromScalar(x interface{}, argMask ...[]bool) ConsOpt {
	var mask []bool
	if len(argMask) > 0 {
		mask = argMask[0]
	}

	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			xt := reflect.TypeOf(x)
			xv := reflect.New(xt)
			xvi := reflect.Indirect(xv)
			xvi.Set(reflect.ValueOf(x))
			ptr := xv.Pointer()
			uptr := unsafe.Pointer(ptr)
			hdr := &reflect.SliceHeader{
				Data: ptr,
				Len:  1,
				Cap:  1,
			}
			tt.data = uptr
			tt.v = x
			tt.t = Dtype{xt}
			tt.hdr = hdr
			tt.mask = mask

		default:
			panic("Unsupported Tensor Type")
		}
	}
	return f
}

// FromMemory is a construction option for creating a *Dense (for now) from memory location. This is a useful
// option for super large tensors that don't fit into memory - the user may need to `mmap` a file the tensor.
//
// Bear in mind that at the current stage of the ConsOpt design, the order of the ConsOpt is important.
// FromMemory  requires the *Dense's Dtype be set already.
// This would fail (and panic):
//		New(FromMemory(ptr, size), Of(Float64))
// This would not:
//		New(Of(Float64), FromMemory(ptr, size))
// This behaviour  of  requiring the ConsOpts to be in order might be changed in the future.
//
// Memory must be manually managed by the caller.
// Tensors called with this construction option will not be returned to any pool - rather, all references to the pointers will be null'd.
// Use with caution.
func FromMemory(ptr uintptr, memsize uintptr) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			tt.v = nil // if there were any underlying slices it should be GC'd
			if tt.hdr == nil {
				tt.hdr = &reflect.SliceHeader{}
			}

			tt.data = unsafe.Pointer(ptr)
			tt.hdr.Data = ptr
			tt.hdr.Len = int(memsize / tt.t.Size())
			tt.hdr.Cap = int(memsize / tt.t.Size())

			tt.flag |= denseFlag(1) << manuallyManagedMem

			switch tt.t {
			case Bool:
				tt.v = *(*[]bool)(unsafe.Pointer(tt.hdr))
			case Int:
				tt.v = *(*[]int)(unsafe.Pointer(tt.hdr))
			case Int8:
				tt.v = *(*[]int8)(unsafe.Pointer(tt.hdr))
			case Int16:
				tt.v = *(*[]int16)(unsafe.Pointer(tt.hdr))
			case Int32:
				tt.v = *(*[]int32)(unsafe.Pointer(tt.hdr))
			case Int64:
				tt.v = *(*[]int64)(unsafe.Pointer(tt.hdr))
			case Uint:
				tt.v = *(*[]uint)(unsafe.Pointer(tt.hdr))
			case Byte:
				tt.v = *(*[]uint8)(unsafe.Pointer(tt.hdr))
			case Uint16:
				tt.v = *(*[]uint16)(unsafe.Pointer(tt.hdr))
			case Uint32:
				tt.v = *(*[]uint32)(unsafe.Pointer(tt.hdr))
			case Uint64:
				tt.v = *(*[]uint64)(unsafe.Pointer(tt.hdr))
			case Float32:
				tt.v = *(*[]float32)(unsafe.Pointer(tt.hdr))
			case Float64:
				tt.v = *(*[]float64)(unsafe.Pointer(tt.hdr))
			case Complex64:
				tt.v = *(*[]complex64)(unsafe.Pointer(tt.hdr))
			case Complex128:
				tt.v = *(*[]complex128)(unsafe.Pointer(tt.hdr))
			case String:
				tt.v = *(*[]string)(unsafe.Pointer(tt.hdr))
			default:
				panic("Unsupported Dtype for using the FromMemory construction option")
			}

		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}
