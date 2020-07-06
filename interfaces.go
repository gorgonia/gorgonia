package gorgonia

import (
	"hash"

	"gorgonia.org/tensor"
)

// Tensor represents values that are acceptable in Gorgonia. At this point, it is implemented by:
// 	- tensor.Tensor
// 	- exprgraph.Node
// 	- *exprgraph.Symbolic
//
// There is an overlap with values.Value. The reason is semantic clarity. Values are Tensors. Tensors are Values.
// There is clearly one issue with this though: *exprgraph.Symbolic.
// *Symbolic is a "fake" tensor, and cannot be lifted as a *dual.Dual.
type Tensor interface {
	// info about the ndarrayN
	Shape() tensor.Shape
	Strides() []int
	Dtype() tensor.Dtype
	Dims() int
	Size() int
	DataSize() int
	Data() interface{}

	// type overloading methods
	IsScalar() bool
	ScalarValue() interface{}

	// engine/memory related stuff
	// all Tensors should be able to be expressed of as a slab of memory
	// Note: the size of each element can be acquired by T.Dtype().Size()
	tensor.Memory
	Engine() tensor.Engine      // Engine can be nil
	IsNativelyAccessible() bool // Can Go access the memory
	IsManuallyManaged() bool    // Must Go manage the memory
}

type hashWriter interface {
	WriteHash(hash.Hash)
}

type arityer interface {
	Arity() int
}

type Tensors []Tensor
