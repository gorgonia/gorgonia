package storage

import (
	"reflect"
	"unsafe"
)

// Header is runtime representation of a slice. It's a cleaner version of reflect.SliceHeader.
// With this, we wouldn't need to keep the uintptr.
// This usually means additional pressure for the GC though, especially when passing around Headers
type Header struct {
	Before [256]byte
	Ptr    unsafe.Pointer
	L      int
	C      int
	After  [256]byte
}

func (h *Header) SlicePtr() unsafe.Pointer { h.Check(); return unsafe.Pointer(&h.Ptr) }
func (h *Header) Pointer() unsafe.Pointer  { h.Check(); return h.Ptr }
func (h *Header) Len() int                 { h.Check(); return h.L }

func check(i int) byte { return byte((i + 1) * 125) }

func (h *Header) Check() {
	if h.Before[0] == 0 {
		for i := range h.Before {
			h.Before[i] = check(i)
			h.After[i] = check(i)
		}
	}

	for i, x := range h.Before {
		if x != check(i) {
			panic("Before smashed")
		}
	}

	for i, x := range h.After {
		if x != check(i) {
			panic("After smashed")
		}
	}
}

func Copy(t reflect.Type, dst, src *Header) int {
	src.Check()
	dst.Check()

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

func CopySliced(t reflect.Type, dst *Header, dstart, dend int, src *Header, sstart, send int) int {
	src.Check()
	dst.Check()

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

func CopyIter(t reflect.Type, dst, src *Header, diter, siter Iterator) int {
	src.Check()
	dst.Check()

	dstBA := AsByteSlice(dst, t)
	srcBA := AsByteSlice(src, t)
	size := int(t.Size())

	var idx, jdx, i, j, count int
	var err error
	for {
		if idx, err = diter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		if jdx, err = siter.Next(); err != nil {
			if err = handleNoOp(err); err != nil {
				panic(err)
			}
			break
		}
		i = idx * size
		j = jdx * size
		copy(dstBA[i:i+size], srcBA[j:j+size])
		// dstBA[i : i+size] = srcBA[j : j+size]
		count++
	}
	return count
}

func AsByteSlice(a *Header, t reflect.Type) []byte {
	a.Check()

	size := a.L * int(t.Size())
	return *(*[]byte)(unsafe.Pointer(&reflect.SliceHeader{
		Data: uintptr(a.Ptr),
		Len:  size,
		Cap:  size,
	}))
}
