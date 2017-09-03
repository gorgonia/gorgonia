// Package tensor is a package that provides efficient, generic n-dimensional arrays in Go.
// Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.
package tensor

import (
	"encoding/gob"
	"fmt"
	"io"
	"unsafe"

	"github.com/pkg/errors"
)

var (
	_ Tensor = &Dense{}
	_ Tensor = &CS{}
)

// Tensor represents a variety of n-dimensional arrays. The most commonly used tensor is the Dense tensor.
// It can be used to represent a vector, matrix, 3D matrix and n-dimensional tensors.
type Tensor interface {
	// info about the ndarray
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int

	// Data access related
	RequiresIterator() bool
	Iterator() Iterator

	// ops
	Slicer
	At(...int) (interface{}, error)
	SetAt(v interface{}, coord ...int) error
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() error // Transpose actually moves the data
	Apply(fn interface{}, opts ...FuncOpt) (Tensor, error)

	// data related interface
	Zeroer
	MemSetter
	Dataer
	Eq
	Cloner

	// type overloading methods
	IsScalar() bool
	ScalarValue() interface{}

	// engine/memory related stuff
	// all Tensors should be able to be expressed of as a slab of memory
	// Note: the size of each element can be acquired by T.Dtype().Size()
	Engine() Engine             // Engine can be nil
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

	headerer
	arrayer
}

// New creates a new Dense Tensor. For sparse arrays use their relevant construction function
func New(opts ...ConsOpt) *Dense {
	d := borrowDense()
	d.AP = new(AP)
	for _, opt := range opts {
		opt(d)
	}
	d.fix()
	if err := d.sanity(); err != nil {
		panic(err)
	}

	return d
}

func assertDense(t Tensor) (*Dense, error) {
	if t == nil {
		return nil, errors.New("nil is not a *Dense")
	}
	if retVal, ok := t.(*Dense); ok {
		return retVal, nil
	}
	return nil, errors.Errorf("%T is not *Dense", t)
}

func getDenseTensor(t Tensor) (DenseTensor, error) {
	switch tt := t.(type) {
	case DenseTensor:
		return tt, nil
	case Densor:
		return tt.Dense(), nil
	default:
		return nil, errors.Errorf("Tensor %T is not a DenseTensor", t)
	}
}

// getFloatDense extracts a *Dense from a Tensor and ensures that the .data is a Array that implements Float
func getFloatDenseTensor(t Tensor) (retVal DenseTensor, err error) {
	if t == nil {
		return
	}
	if err = typeclassCheck(t.Dtype(), floatTypes); err != nil {
		err = errors.Wrapf(err, "getFloatDense only handles floats. Got %v instead", t.Dtype())
		return
	}

	if retVal, err = getDenseTensor(t); err != nil {
		err = errors.Wrapf(err, opFail, "getFloatDense")
		return
	}
	if retVal == nil {
		return
	}

	return
}

func sliceDense(t *Dense, slices ...Slice) (retVal *Dense, err error) {
	var sliced Tensor
	if sliced, err = t.Slice(slices...); err != nil {
		return nil, err
	}
	return sliced.(*Dense), nil
}
