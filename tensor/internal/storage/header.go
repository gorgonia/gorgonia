package storage

import (
	"reflect"
	"unsafe"
)

// Header is runtime representation of a slice. It's a cleaner version of reflect.SliceHeader.
// With this, we wouldn't need to keep the uintptr.
// This usually means additional pressure for the GC though, especially when passing around Headers
type Header struct {
	Ptr unsafe.Pointer
	L   int
	C   int
}

func (h *Header) Pointer() unsafe.Pointer { return h.Ptr }
func (h *Header) Len() int                { return h.L }

func Copy(dst, src *Header, t reflect.Type) int {
	if dst.L == 0 || src.L == 0 {
		return 0
	}

	n := src.L
	if dst.L < n {
		n = dst.L
	}

	// handle struct{} type
	if t.Size() == 0 {
		return n
	}

	// memmove(dst.Pointer(), src.Pointer(), t.Size())
	// return n

	// otherwise, just copy bytes.
	// FUTURE: implement memmove
	dstBA := AsByteSlice(dst, t)
	srcBA := AsByteSlice(src, t)
	copied := copy(dstBA, srcBA)
	return copied / int(t.Size())
}

func CopySliced(dst *Header, dstart, dend int, src *Header, sstart, send int, t reflect.Type) int {
	dstBA := AsByteSlice(dst, t)
	srcBA := AsByteSlice(src, t)
	size := int(t.Size())
	ds := dstart * size
	de := dend * size
	ss := sstart * size
	se := send * size
	copied := copy(dstBA[ds:de], srcBA[ss:se])
	return copied / size
}

func AsByteSlice(a *Header, t reflect.Type) []byte {
	size := a.L * int(t.Size())
	hdr := reflect.SliceHeader{
		Data: uintptr(a.Ptr),
		Len:  size,
		Cap:  size,
	}
	return *(*[]byte)(unsafe.Pointer(&hdr))
}
