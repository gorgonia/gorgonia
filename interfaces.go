package gorgonia

import (
	"hash"
	"unsafe"

	"gorgonia.org/tensor"
)

// Tensor is an interface that describes an ndarray
type Tensor interface {
	// info about the ndarrayN
	Shape() tensor.Shape
	Strides() []int
	Dtype() tensor.Dtype
	Dims() int
	Size() int
	DataSize() int

	// type overloading methods
	IsScalar() bool
	ScalarValue() interface{}

	// engine/memory related stuff
	// all Tensors should be able to be expressed of as a slab of memory
	// Note: the size of each element can be acquired by T.Dtype().Size()
	Engine() tensor.Engine      // Engine can be nil
	MemSize() uintptr           // the size in memory
	Uintptr() uintptr           // the pointer to the first element, as a uintptr
	Pointer() unsafe.Pointer    // the pointer to the first elemment as a unsafe.Ponter
	IsNativelyAccessible() bool // Can Go access the memory
	IsManuallyManaged() bool    // Must Go manage the memory
}

type hashWriter interface {
	WriteHash(hash.Hash)
}

type arityer interface {
	Arity() int
}
