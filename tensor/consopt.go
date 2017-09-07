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
		case *CS:
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
				tt.addMask(mask)
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
		case *CS:
			if len(dims) != 2 {
				panic("Only sparse matrices are supported")
			}
			throw := BorrowInts(len(dims))
			copy(throw, dims)
			tt.s = throw

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

			tt.array.Ptr = uptr
			tt.array.L = 1
			tt.array.C = 1
			tt.v = x
			tt.t = Dtype{xt}
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

			tt.array.Ptr = unsafe.Pointer(ptr)
			tt.array.L = int(memsize / tt.t.Size())
			tt.array.C = int(memsize / tt.t.Size())

			tt.flag = MakeMemoryFlag(tt.flag, ManuallyManaged)

			if tt.IsNativelyAccessible() {
				tt.array.fix()
			}

		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithEngine is a construction option that would cause a Tensor to be linked with an execution engine.
func WithEngine(e Engine) ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			tt.e = e
			if e != nil && !e.AllocAccessible() {
				tt.flag = MakeMemoryFlag(tt.flag, NativelyInaccessible)
			}
			// if oe, ok := e.(standardEngine); ok {
			// 	tt.oe = oe
			// }
		case *CS:
			tt.e = e
			if e != nil && !e.AllocAccessible() {
				tt.f = MakeMemoryFlag(tt.f, NativelyInaccessible)
			}
		}
	}
	return f
}

func AsFortran() ConsOpt {
	f := func(t Tensor) {
		switch tt := t.(type) {
		case *Dense:
			if tt.AP == nil {
				// create AP
			}
			tt.AP.o = MakeDataOrder(tt.AP.o, ColMajor)
		}
	}
	return f
}
