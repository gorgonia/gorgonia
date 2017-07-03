package tensor

import (
	"fmt"
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

// array is the underlying generic array.
type array struct {
	header             // the header - the Go representation (a slice)
	t      Dtype       // the element type
	v      interface{} // an additional reference to the underlying slice. This is not strictly necessary, but does improve upon anything that calls .Data()
}

// makeArray makes an array. The memory allocation is handled by Go
func makeArray(t Dtype, length int) array {
	hdr := makeHeader(t, length)
	return makeArrayFromHeader(hdr, t)
}

// makeArrayFromHeader makes an array given a header
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

// arrayFromSlice creates an array from a slice. If x is not a slice, it will panic.
func arrayFromSlice(x interface{}) array {
	xT := reflect.TypeOf(x)
	if xT.Kind() != reflect.Slice {
		panic("Expected a slice")
	}
	elT := xT.Elem()

	xV := reflect.ValueOf(x)
	ptr := xV.Pointer()
	uptr := unsafe.Pointer(ptr)

	return array{
		header: header{
			ptr: uptr,
			l:   xV.Len(),
			c:   xV.Cap(),
		},
		t: Dtype{elT},
		v: x,
	}
}

// byteSlice casts the underlying slice into a byte slice. Useful for copying and zeroing, but not much else
func (a array) byteSlice() []byte {
	size := a.l * int(a.t.Size())
	hdr := reflect.SliceHeader{
		Data: uintptr(a.ptr),
		Len:  size,
		Cap:  size,
	}
	return *(*[]byte)(unsafe.Pointer(&hdr))
}

// sliceInto creates a slice. Instead of returning an array, which would cause a lot of reallocations, sliceInto expects a array to
// already have been created. This allows repetitive actions to be done without having to have many pointless allocation
func (a array) sliceInto(i, j int, res *array) {
	base := uintptr(a.ptr)
	c := a.c

	if i < 0 || j < i || j > c {
		panic(fmt.Sprintf("Cannot slice %v - index %d:%d is out of bounds", a, i, j))
	}

	res.l = j - i
	res.c = c - i

	if c-1 > 0 {
		res.ptr = unsafe.Pointer(base + uintptr(i)*a.t.Size())
	} else {
		// don't adviance
		res.ptr = unsafe.Pointer(base)
	}
}

// swap swaps the elements i and j in the array
func (a array) swap(i, j int) {
	if a.t == String {
		ss := *(*[]string)(a.ptr)
		ss[i], ss[j] = ss[j], ss[i]
		return
	}
	if !isParameterizedKind(a.t.Kind()) {
		switch a.t.Size() {
		case 8:
			us := *(*[]uint64)(unsafe.Pointer(&a.header))
			us[i], us[j] = us[j], us[i]
		case 4:
			us := *(*[]uint32)(unsafe.Pointer(&a.header))
			us[i], us[j] = us[j], us[i]
		case 2:
			us := *(*[]uint16)(unsafe.Pointer(&a.header))
			us[i], us[j] = us[j], us[i]
		case 1:
			us := *(*[]uint8)(unsafe.Pointer(&a.header))
			us[i], us[j] = us[j], us[i]
		}
		return
	}

	tmp := make([]byte, a.t.Size())
	bs := a.byteSlice()
	is := i * a.t.Size()
	ie := is + a.t.Size()
	js := j * a.t.Size()
	je := js + a.t.Size()
	copy(tmp, bs[is:ie])
	copy(bs[is:ie], bs[js:je])
	copy(bs[js:je], tmp)
}

// Data returns the representation of a slice.
func (a array) Data() interface{} { return a.v }

// Zero zeroes out the underlying array of the *Dense tensor.
func (a array) Zero() {
	if !isParameterizedKind(a.t.Kind()) {
		ba := a.byteSlice()
		for i := range ba {
			ba[i] = 0
		}
		return
	}
	if a.t.Kind() == reflect.String {
		ss := a.strings()
		for i := range ss {
			ss[i] = ""
		}
		return
	}
	ptr := uintptr(a.ptr)
	for i := 0; i < a.l; i++ {
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(reflect.Zero(a.t))
	}
}

// copyArray copies an array.
func copyArray(dst, src array) int {
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}

	if dst.l == 0 || src.l == 0 {
		return 0
	}

	n := src.l
	if dst.l < n {
		n = dst.l
	}

	// handle struct{} type
	if dst.t.Size() == 0 {
		return n
	}

	// otherwise, just copy bytes.
	// FUTURE: implement memmove
	dstBA := dst.byteSlice()
	srcBA := src.byteSlice()
	return copy(dstBA, srcBA)
}
