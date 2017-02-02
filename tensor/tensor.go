package tensor

import (
	"fmt"
	"reflect"

	"github.com/pkg/errors"
)

type Tensor interface {
	// info about the ndarray
	Info() *AP
	Shape() Shape
	Strides() []int
	Dtype() Dtype
	Dims() int
	Size() int
	DataSize() int

	// ops
	At(...int) (interface{}, error)
	Reshape(...int) error
	T(axes ...int) error
	UT()
	Transpose() // Transpose actually moves the data
	Slice(...Slice) (Tensor, error)

	// data related interface
	Zeroer
	MemSetter
	Dataer
	Eq
	Cloner

	// type overloading shit
	IsScalar() bool
	ScalarValue() interface{}

	// view related shit
	IsView() bool
	Materialize() Tensor

	// formatters
	fmt.Formatter
	fmt.Stringer

	// all Tensors are serializable to these formats
	// WriteNpy(io.Writer) error
	// ReadNpy(io.Reader) error
	// gob.GobEncoder
	// gob.GobDecoder
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
