// Package tensor is a package that provides efficient, generic n-dimensional arrays in Go.
// Also in this package are functions and methods that are used commonly in arithmetic, comparison and linear algebra operations.
package tensor

import (
	"encoding/gob"
	"fmt"
	"io"
	"reflect"
	"unsafe"

	"github.com/pkg/errors"
)

// Tensor represents a variety of n-dimensional arrays. The most commonly used tensor is the Dense tensor.
// It can be used to represent a vector, matrix, 3D matrix and n-dimensional tensors.
type Tensor interface {
	// info about the ndarray
	Info() *AP
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int
	Engine() Engine // Engine can be nil

	// ops
	At(...int) (interface{}, error)
	SetAt(v interface{}, coord ...int) error
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() // Transpose actually moves the data
	Slice(...Slice) (Tensor, error)
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

	// view related methods
	IsView() bool
	Materialize() Tensor

	// all Tensors should be able to be expressed of as a slab of memory
	// Note: the size of each element can be acquired by T.Dtype().Size()
	MemSize() uintptr        // the size in memory
	Uintptr() uintptr        // the pointer to the first element, as a uintptr
	Pointer() unsafe.Pointer // the pointer to the first elemment as a unsafe.Ponter

	// formatters
	fmt.Formatter
	fmt.Stringer

	// all Tensors are serializable to these formats
	WriteNpy(io.Writer) error
	ReadNpy(io.Reader) error
	gob.GobEncoder
	gob.GobDecoder
}

// Dotter is used to implement sparse matrices
type Dotter interface {
	Tensor
	Dot(Tensor, ...FuncOpt) (Tensor, error)
}

// New creates a new Dense Tensor. For sparse arrays use their relevant construction function
func New(opts ...ConsOpt) *Dense {
	d := new(Dense)
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

func getDense(t Tensor) (*Dense, error) {
	if t == nil {
		return nil, nil
	}

	if retVal, ok := t.(*Dense); ok {
		return retVal, nil
	}

	// TODO: when sparse matrices are created, add a clause here to return early

	if densor, ok := t.(Densor); ok {
		return densor.Dense(), nil
	}

	v := reflect.ValueOf(t)
	tt := reflect.TypeOf(t)
	var s, zero reflect.Value
	if tt.Kind() == reflect.Ptr {
		of := tt.Elem()
		if of.Kind() != reflect.Struct {
			return nil, errors.Errorf("Cannot extract *Dense from %v of %T", t, t)
		}
		s = v.Elem()
	} else if tt.Kind() == reflect.Struct {
		s = v
	}
	d := s.FieldByName("Dense")
	if d != zero {
		return d.Interface().(*Dense), nil
	}

	return nil, errors.Errorf(extractionFail, "*Dense", t)
}

// getFloatDense extracts a *Dense from a Tensor and ensures that the .data is a Array that implements Float
func getFloatDense(t Tensor) (retVal *Dense, err error) {
	if t == nil {
		return
	}
	if retVal, err = getDense(t); err != nil {
		err = errors.Wrapf(err, opFail, "getFloatDense")
		return
	}
	if retVal == nil {
		return
	}
	if !isFloat(retVal.t) {
		err = errors.Errorf(dtypeMismatch, retVal.t, retVal.data)
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
