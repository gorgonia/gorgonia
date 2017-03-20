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
// It can also support creation of masked arrays via optional argument argMask
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
			tt.fromSlice(x, mask)
		default:
			panic("Unsupported Tensor type")
		}
	}
	return f
}

// WithShape is a construction option for a Tensor. It creates the ndarray in the required shape
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
