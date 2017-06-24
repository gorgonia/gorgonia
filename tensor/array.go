package tensor

import (
	"reflect"
	"unsafe"
)

// header is like reflect.SliceHeader. It is used to do very dirty dirty things.
type header struct {
	ptr unsafe.Pointer
	l   int
	c   int
}

func makeHeader(t Dtype, length int) header {
	size := int(calcMemSize(t, length))
	s := make([]byte, size)
	return header{
		ptr: unsafe.Pointer(&s[0]),
		l:   length,
		c:   length,
	}
}

// array is the underlying generic array
type array struct {
	header             // the header - the Go representation (a slice)
	t      Dtype       // the element type
	v      interface{} // an additional reference to the underlying slice
}

// makeArray makes an array. The memory allocation is handled by Go
func makeArray(t Dtype, length int) array {
	hdr := makeHeader(t, length)
	return makeArrayFromHeader(hdr, t)
}

func makeArrayFromHeader(hdr header, t Dtype) array {
	// build a type of []T
	shdr := reflect.SliceHeader{
		Data: uintptr(hdr.ptr),
		Len:  hdr.l,
		Cap:  hdr.l,
	}
	sliceT := reflect.SliceOf(t.Type)
	ptr := unsafe.Pointer(&shdr)
	val := reflect.Indirect(reflect.NewAt(sliceT, ptr))

	return array{
		header: hdr,
		t:      t,
		v:      val.Interface(),
	}
}

// func copyArray(dst, src array) int {
// 	if dst.t != src.t {
// 		panic("Cannot copy arrays of different types.")
// 	}

// 	if dst.l == 0 || src.l == 0 {
// 		return 0
// 	}

// 	// handle struct{} type
// 	if dst.t.Size() == 0 {
// 		n := src.l
// 		if dst.l < n {
// 			n = dst.l
// 		}
// 		return n
// 	}

// 	// otherwise, just copy bytes.
// 	// FUTURE: implement memmove
// 	dstBA := dst.uint8s()
// 	srcBA := src.uint8s()
// 	return copy(dstBA, srcBA)
// }
