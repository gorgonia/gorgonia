package stdeng

import (
	"reflect"
	"unsafe"
)

// header is runtime representation of a slice. It's a cleaner version of reflect.SliceHeader.
// With this, we wouldn't need to keep the uintptr.
// This usually means additional pressure for the GC though, especially when passing around headers
type header struct {
	ptr unsafe.Pointer
	l   int
	c   int
}

func arrayFromSlice(x interface{}) *header {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	// elT := xT.Elem()

	xV := reflect.ValueOf(x)
	ptr := xV.Pointer()
	uptr := unsafe.Pointer(ptr)
	return &header{
		ptr: uptr,
		l:   xV.Len(),
		c:   xV.Cap(),
	}
}
