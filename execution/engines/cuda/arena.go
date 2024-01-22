package cuda

import (
	"unsafe"

	"gorgonia.org/cu"
	"gorgonia.org/tensor"
)

// this file implements an Arena.

var _ tensor.Memory = cu.DevicePtr(0)

type hostmemory interface {
	tensor.Memory
	IsHostMemory() bool
}

type cudamemory struct {
	ptr cu.DevicePtr
	sz  uintptr
}

func (mem cudamemory) Uintptr() uintptr { return mem.ptr.Uintptr() }
func (mem cudamemory) MemSize() uintptr { return mem.sz }

// Get allocates memory of certain size and returns a pointer to it
func (e *Engine[DT, T]) Get(size int64) (tensor.Memory, error) {
	ptr, err := e.a.Alloc(size)
	return cudamemory{cu.DevicePtr(ptr), uintptr(size)}, err
}

// Put releases a chunk of memory of certain size
func (e *Engine[DT, T]) Put(mem tensor.Memory, size int64) { e.a.Free(mem.Uintptr()) }

// ResetAllocator releases used memory of the engine
func (e *Engine[DT, T]) ResetAllocator() { e.a.Reset() }

func (e *Engine[DT, T]) memcpy(dst, src tensor.Memory, sz int64) {
	switch d := dst.(type) {
	case cu.DevicePtr:
		switch s := src.(type) {
		case cu.DevicePtr:
			e.c.Memcpy(d, s, sz)
			return
		case cudamemory:
			e.c.Memcpy(d, s.ptr, sz)
			return
		case hostmemory:
			e.c.MemcpyHtoD(d, unsafe.Pointer(s.Uintptr()), sz)
			return
		case tensor.DescWithStorage:
			f := s.Flags()
			if f.IsNativelyAccessible() {
				e.c.MemcpyHtoD(d, unsafe.Pointer(s.Uintptr()), sz)
				return
			}
			e.c.Memcpy(d, cu.DevicePtr(s.Uintptr()), sz)
			return
		}
	case cudamemory:
		switch s := src.(type) {
		case cu.DevicePtr:
			e.c.Memcpy(d.ptr, s, sz)
			return
		case cudamemory:
			e.c.Memcpy(d.ptr, s.ptr, sz)
			return
		case hostmemory:
			e.c.MemcpyHtoD(d.ptr, unsafe.Pointer(s.Uintptr()), sz)
			return
		case tensor.DescWithStorage:
			f := s.Flags()
			if f.IsNativelyAccessible() {
				e.c.MemcpyHtoD(d.ptr, unsafe.Pointer(s.Uintptr()), sz)
				return
			}
			e.c.Memcpy(d.ptr, cu.DevicePtr(s.Uintptr()), sz)
			return
		}

	case hostmemory:
		switch s := src.(type) {
		case cu.DevicePtr:
			e.c.MemcpyDtoH(unsafe.Pointer(d.Uintptr()), s, sz)
			return
		case cudamemory:
			e.c.MemcpyDtoH(unsafe.Pointer(d.Uintptr()), s.ptr, sz)
			return
		case hostmemory:
			panic("NYI")
		case tensor.DescWithStorage:
			f := s.Flags()
			if f.IsNativelyAccessible() {
				panic("NYI")
			}
			e.c.MemcpyDtoH(unsafe.Pointer(d.Uintptr()), cu.DevicePtr(s.Uintptr()), sz)
			return
		}
	case tensor.DescWithStorage:
		f := d.Flags()

		switch s := src.(type) {
		case cu.DevicePtr:
			if f.IsNativelyAccessible() {
				e.c.MemcpyDtoH(unsafe.Pointer(d.Uintptr()), s, sz)
				return
			}
			e.c.Memcpy(cu.DevicePtr(d.Uintptr()), s, sz)
			return
		case hostmemory:
			panic("NYI")

		}
	}

	panic("NYI")
}
