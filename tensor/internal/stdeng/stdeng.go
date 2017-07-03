package stdeng

import (
	"reflect"
	"unsafe"
)

type Array interface {
	Bools() []bool
	Complex128s() []complex128
	Complex64s() []complex64
	Float32s() []float32
	Float64s() []float64
	Int16s() []int16
	Int32s() []int32
	Int64s() []int64
	Int8s() []int8
	Ints() []int
	Strings() []string
	Uint16s() []uint16
	Uint32s() []uint32
	Uint64s() []uint64
	Uint8s() []uint8
	Uintptrs() []uintptr
	Uints() []uint
	UnsafePointers() []unsafe.Pointer

	Pointer() unsafe.Pointer
	Len() int
}

// Iterator is the generic iterator interface
type Iterator interface {
	Start() (int, error)
	Next() (int, error)
	NextValid() (int, int, error)
	NextInvalid() (int, int, error)
	Reset()
	SetReverse()
	SetForward()
	Coord() []int
	Done() bool
}

// E is the standard engine. It's to be embedded in package tensor
type E struct{}

func (e E) Add(a, b Array, t reflect.Type)        {}
func (e E) AddIter(a, b Array, ait, bit Iterator) {}
