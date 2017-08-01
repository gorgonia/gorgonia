package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
)

// array is the underlying generic array.
type array struct {
	storage.Header             // the header - the Go representation (a slice)
	t              Dtype       // the element type
	v              interface{} // an additional reference to the underlying slice. This is not strictly necessary, but does improve upon anything that calls .Data()
}

// makeArray makes an array. The memory allocation is handled by Go
func makeArray(t Dtype, length int) array {
	hdr := makeHeader(t, length)
	return makeArrayFromHeader(hdr, t)
}

// makeArrayFromHeader makes an array given a header
func makeArrayFromHeader(hdr storage.Header, t Dtype) array {
	// build a type of []T
	shdr := reflect.SliceHeader{
		Data: uintptr(hdr.Ptr),
		Len:  hdr.L,
		Cap:  hdr.L,
	}
	sliceT := reflect.SliceOf(t.Type)
	ptr := unsafe.Pointer(&shdr)
	val := reflect.Indirect(reflect.NewAt(sliceT, ptr))

	return array{
		Header: hdr,
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
		Header: storage.Header{
			Ptr: uptr,
			L:   xV.Len(),
			C:   xV.Cap(),
		},
		t: Dtype{elT},
		v: x,
	}
}

// byteSlice casts the underlying slice into a byte slice. Useful for copying and zeroing, but not much else
func (a array) byteSlice() []byte {
	return storage.AsByteSlice(&a.Header, a.t.Type)
}

// sliceInto creates a slice. Instead of returning an array, which would cause a lot of reallocations, sliceInto expects a array to
// already have been created. This allows repetitive actions to be done without having to have many pointless allocation
func (a array) sliceInto(i, j int, res *array) {
	base := uintptr(a.Ptr)
	c := a.C

	if i < 0 || j < i || j > c {
		panic(fmt.Sprintf("Cannot slice %v - index %d:%d is out of bounds", a, i, j))
	}

	res.L = j - i
	res.C = c - i

	if c-1 > 0 {
		res.Ptr = unsafe.Pointer(base + uintptr(i)*a.t.Size())
	} else {
		// don't adviance
		res.Ptr = unsafe.Pointer(base)
	}
}

// swap swaps the elements i and j in the array
func (a array) swap(i, j int) {
	if a.t == String {
		ss := *(*[]string)(a.Ptr)
		ss[i], ss[j] = ss[j], ss[i]
		return
	}
	if !isParameterizedKind(a.t.Kind()) {
		switch a.t.Size() {
		case 8:
			us := *(*[]uint64)(unsafe.Pointer(&a.Header))
			us[i], us[j] = us[j], us[i]
		case 4:
			us := *(*[]uint32)(unsafe.Pointer(&a.Header))
			us[i], us[j] = us[j], us[i]
		case 2:
			us := *(*[]uint16)(unsafe.Pointer(&a.Header))
			us[i], us[j] = us[j], us[i]
		case 1:
			us := *(*[]uint8)(unsafe.Pointer(&a.Header))
			us[i], us[j] = us[j], us[i]
		}
		return
	}

	size := int(a.t.Size())
	tmp := make([]byte, size)
	bs := a.byteSlice()
	is := i * size
	ie := is + size
	js := j * size
	je := js + size
	copy(tmp, bs[is:ie])
	copy(bs[is:ie], bs[js:je])
	copy(bs[js:je], tmp)
}

/* *Array is a Memory */

// Uintptr returns the pointer of the first value of the slab
func (t array) Uintptr() uintptr { return uintptr(t.Ptr) }

// MemSize returns how big the slice is in bytes
func (t array) MemSize() uintptr { return uintptr(t.L) * t.t.Size() }

// Pointer returns the pointer of the first value of the slab, as an unsafe.Pointer
func (t array) Pointer() unsafe.Pointer { return t.Ptr }

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
		ss := a.Strings()
		for i := range ss {
			ss[i] = ""
		}
		return
	}
	ptr := uintptr(a.Ptr)
	for i := 0; i < a.L; i++ {
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(reflect.Zero(a.t))
	}
}

func (a array) hdr() *storage.Header { return &a.Header }
func (a array) rtype() reflect.Type  { return a.t.Type }

// copyArray copies an array.
func copyArray(dst, src array) int {
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}
	return storage.Copy(dst.t.Type, &dst.Header, &src.Header)
}

// copyDense copies a DenseTensor
func copyDense(dest, src DenseTensor) int {
	e := src.Engine()
	switch s := src.(type) {
	// case *MaskedDense:
	// var d *MaskedDense
	// var ok bool
	// if d, ok = dest.(*MaskedDense); !ok {
	// 	panic("Copy can only copy from *MaskedDense to *MaskedDense")
	// }

	// if s.IsMasked() {
	// 	if cap(d.mask) < len(s.mask) {
	// 		d.mask = make([]bool, len(s.mask))
	// 	}
	// 	copy(d.mask, s.mask)
	// 	d.mask = d.mask[:len(s.mask)]
	// }
	// if e != nil {
	// 	if err := e.Memcpy(d.array, s.array); err != nil {
	// 		panic(err)
	// 	}
	// 	return d.L
	// }
	// return copyArray(d.array, s.array)

	case *Dense:
		var d *Dense
		var ok bool
		if d, ok = dest.(*Dense); !ok {
			panic("Copy can only copy from *Dense to *Dense")
		}
		if e != nil {
			if err := e.Memcpy(d.array, s.array); err != nil {
				panic(err)
			}
			return d.L
		}
		return copyArray(d.array, s.array)
	}
}
