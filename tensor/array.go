package tensor

import (
	"fmt"
	"reflect"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor/internal/storage"
	"github.com/pkg/errors"
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
	hdr.Check()
	return makeArrayFromHeader(hdr, t)
}

// makeArrayFromHeader makes an array given a header
func makeArrayFromHeader(hdr storage.Header, t Dtype) array {
	hdr.Check()

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

	arr := array{
		Header: storage.Header{
			Ptr: uptr,
			L:   xV.Len(),
			C:   xV.Cap(),
		},
		t: Dtype{elT},
		v: x,
	}
	arr.Check()
	return arr
}

// byteSlice casts the underlying slice into a byte slice. Useful for copying and zeroing, but not much else
func (a array) byteSlice() []byte {
	a.Check()
	return storage.AsByteSlice(&a.Header, a.t.Type)
}

// sliceInto creates a slice. Instead of returning an array, which would cause a lot of reallocations, sliceInto expects a array to
// already have been created. This allows repetitive actions to be done without having to have many pointless allocation
func (a array) sliceInto(i, j int, res *array) {
	a.Check()
	res.Check()

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

func (a array) slice(start, end int) array {
	size := int(a.t.Size())
	if end > a.L {
		panic("Index out of range")
	}
	startptr := unsafe.Pointer(uintptr(a.Pointer()) + uintptr(start*size))

	arr := array{
		Header: storage.Header{
			Ptr: startptr,
			L:   end - start + 1,
			C:   end - start + 1,
		},
		t: a.t,
	}
	arr.Check()
	return arr
}

// swap swaps the elements i and j in the array
func (a array) swap(i, j int) {
	a.Check()
	if a.t == String {
		ss := *(*[]string)(a.Ptr)
		ss[i], ss[j] = ss[j], ss[i]
		return
	}
	if !isParameterizedKind(a.t.Kind()) {
		switch a.t.Size() {
		case 8:
			us := *(*[]uint64)(a.SlicePtr())
			us[i], us[j] = us[j], us[i]
		case 4:
			us := *(*[]uint32)(a.SlicePtr())
			us[i], us[j] = us[j], us[i]
		case 2:
			us := *(*[]uint16)(a.SlicePtr())
			us[i], us[j] = us[j], us[i]
		case 1:
			us := *(*[]uint8)(a.SlicePtr())
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
func (t array) Uintptr() uintptr { t.Check(); return uintptr(t.Ptr) }

// MemSize returns how big the slice is in bytes
func (t array) MemSize() uintptr { t.Check(); return uintptr(t.L) * t.t.Size() }

// Pointer returns the pointer of the first value of the slab, as an unsafe.Pointer
func (t array) Pointer() unsafe.Pointer { t.Check(); return t.Ptr }

// Data returns the representation of a slice.
func (a array) Data() interface{} { a.Check(); return a.v }

// Zero zeroes out the underlying array of the *Dense tensor.
func (a array) Zero() {
	a.Check()
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

func (a array) hdr() *storage.Header { a.Header.Check(); return &a.Header }
func (a array) rtype() reflect.Type  { return a.t.Type }

// copyArray copies an array.
func copyArray(dst, src array) int {
	src.Check()
	dst.Check()
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}
	return storage.Copy(dst.t.Type, &dst.Header, &src.Header)
}

func copyArraySliced(dst array, dstart, dend int, src array, sstart, send int) int {
	src.Check()
	dst.Check()
	if dst.t != src.t {
		panic("Cannot copy arrays of different types.")
	}
	return storage.CopySliced(dst.t.Type, &dst.Header, dstart, dend, &src.Header, sstart, send)
}

// copyDense copies a DenseTensor
func copyDense(dst, src DenseTensor) int {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot dopy DenseTensors of different types")
	}

	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < len(smask) {
				dmask = make([]bool, len(smask))
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask, smask)
		}
	}

	e := src.Engine()
	if e != nil {
		if err := e.Memcpy(dst.arr(), src.arr()); err != nil {
			panic(err)
		}
		return dst.len()
	}
	return copyArray(dst.arr(), src.arr())
}

func copyDenseSliced(dst DenseTensor, dstart, dend int, src DenseTensor, sstart, send int) int {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot copy DenseTensors of different types")
	}

	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < dend {
				dmask = make([]bool, dend)
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask[dstart:dend], smask[sstart:send])
		}
	}
	if e := src.Engine(); e != nil {
		d := dst.arr().slice(dstart, dend)
		s := src.arr().slice(sstart, send)
		if err := e.Memcpy(d, s); err != nil {
			panic(err)
		}
		return d.Len()
	}
	return copyArraySliced(dst.arr(), dstart, dend, src.arr(), sstart, send)
}

func copyDenseIter(dst, src DenseTensor, diter, siter Iterator) (int, error) {
	if dst.Dtype() != src.Dtype() {
		panic("Cannot copy Dense arrays of different types")
	}
	if !requiresIterator(dst) && !requiresIterator(src) {
		return copyDense(dst, src), nil
	}
	if !dst.IsNativelyAccessible() || !src.IsNativelyAccessible() {
		return 0, errors.Errorf(inaccessibleData, "copy")
	}

	if diter == nil {
		diter = FlatIteratorFromDense(dst)
	}
	if siter == nil {
		siter = FlatIteratorFromDense(src)
	}

	if ms, ok := src.(MaskedTensor); ok && ms.IsMasked() {
		if md, ok := dst.(MaskedTensor); ok {
			dmask := md.Mask()
			smask := ms.Mask()
			if cap(dmask) < len(smask) {
				dmask = make([]bool, len(smask))
				copy(dmask, md.Mask())
				md.SetMask(dmask)
			}
			copy(dmask, smask)
		}
	}
	return storage.CopyIter(dst.rtype(), dst.hdr(), src.hdr(), diter, siter), nil
}
