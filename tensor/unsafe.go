package tensor

import (
	"reflect"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/stdeng"
)

var _ stdeng.Array = &header{}

// header is runtime representation of a slice. It's a cleaner version of reflect.SliceHeader.
// With this, we wouldn't need to keep the uintptr.
// This usually means additional pressure for the GC though, especially when passing around headers
type header struct {
	ptr unsafe.Pointer
	l   int
	c   int
}

// makeHeader makes a array header
func makeHeader(t Dtype, length int) header {
	size := int(calcMemSize(t, length))
	s := make([]byte, size)
	return header{
		ptr: unsafe.Pointer(&s[0]),
		l:   length,
		c:   length,
	}
}

func (h *header) Pointer() unsafe.Pointer { return h.ptr }
func (h *header) Len() int                { return h.l }

func copyHeader(dst, src *header, t reflect.Type) int {
	if dst.l == 0 || src.l == 0 {
		return 0
	}

	n := src.l
	if dst.l < n {
		n = dst.l
	}

	// handle struct{} type
	if t.Size() == 0 {
		return n
	}

	// memmove(dst.Pointer(), src.Pointer(), t.Size())
	// return n

	// otherwise, just copy bytes.
	// FUTURE: implement memmove
	dstBA := asByteSlice(dst, t)
	srcBA := asByteSlice(src, t)
	copied := copy(dstBA, srcBA)
	return copied / int(t.Size())
}

func asByteSlice(a *header, t reflect.Type) []byte {
	size := a.l * int(t.Size())
	hdr := reflect.SliceHeader{
		Data: uintptr(a.ptr),
		Len:  size,
		Cap:  size,
	}
	return *(*[]byte)(unsafe.Pointer(&hdr))
}

// iface is equivalent to a truncated version of reflect.Value. It's here for the quick extraction of the data pointer
// without having to use the reflect.ValueOf which is slower.
//
// The downside is of course, having to manually keepo up with the development of Go's language internals
type iface [2]uintptr

func extractPointer(a interface{}) unsafe.Pointer {
	return unsafe.Pointer((*[2]uintptr)(unsafe.Pointer(&a))[1])
}

func scalarToHeader(a interface{}) *header {
	return &header{
		ptr: extractPointer(a),
		l:   1,
		c:   1,
	}
}

/* REALLY REALLY UNSAFE STUFF. */

//go:linkname memmove runtime.memmove
func memmove(to, from unsafe.Pointer, n uintptr)
