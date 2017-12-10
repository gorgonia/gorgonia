package gorgonia

import (
	"encoding/gob"
	"fmt"
	"io"
	"unsafe"

	"gorgonia.org/tensor"
)

type Tensor interface {
	// info about the ndarray
	Shape() tensor.Shape
	Strides() []int
	Dtype() tensor.Dtype
	Dims() int
	Size() int
	DataSize() int

	// Data access related
	RequiresIterator() bool
	Iterator() tensor.Iterator

	// ops
	tensor.Slicer
	At(...int) (interface{}, error)
	SetAt(v interface{}, coord ...int) error
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() error // Transpose actually moves the data
	Apply(fn interface{}, opts ...tensor.FuncOpt) (tensor.Tensor, error)

	// data related interface
	tensor.Zeroer
	tensor.MemSetter
	tensor.Dataer
	tensor.Eq
	tensor.Cloner

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

	// formatters
	fmt.Formatter
	fmt.Stringer

	// all Tensors are serializable to these formats
	WriteNpy(io.Writer) error
	ReadNpy(io.Reader) error
	gob.GobEncoder
	gob.GobDecoder
}
