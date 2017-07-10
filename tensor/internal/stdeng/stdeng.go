package stdeng

import (
	"reflect"
	"unsafe"
)

// Array is an interface representing the slice header. It can be cast into various slice types
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

func isScalar(a Array) bool { return a.Len() == 1 }

// Iterator is the generic iterator interface
type Iterator interface {
	Start() (int, error)
	Next() (int, error)
	NextValidity() (int, bool, error)
	NextValid() (int, int, error)
	NextInvalid() (int, int, error)
	Reset()
	SetReverse()
	SetForward()
	Coord() []int
	Done() bool
}

// basic types supported.
var (
	Bool       = reflect.TypeOf(true)
	Int        = reflect.TypeOf(int(1))
	Int8       = reflect.TypeOf(int8(1))
	Int16      = reflect.TypeOf(int16(1))
	Int32      = reflect.TypeOf(int32(1))
	Int64      = reflect.TypeOf(int64(1))
	Uint       = reflect.TypeOf(uint(1))
	Uint8      = reflect.TypeOf(uint8(1))
	Uint16     = reflect.TypeOf(uint16(1))
	Uint32     = reflect.TypeOf(uint32(1))
	Uint64     = reflect.TypeOf(uint64(1))
	Float32    = reflect.TypeOf(float32(1))
	Float64    = reflect.TypeOf(float64(1))
	Complex64  = reflect.TypeOf(complex64(1))
	Complex128 = reflect.TypeOf(complex128(1))
	String     = reflect.TypeOf("")

	// aliases
	Byte = Uint8

	// extras
	Uintptr       = reflect.TypeOf(uintptr(0))
	UnsafePointer = reflect.TypeOf(unsafe.Pointer(&Uintptr))
)

// E is the standard engine. It's to be embedded in package tensor
type E struct{}

// Op performs the operation.
func (e E) Op(t reflect.Type, a, b Array) error { return nil }

// OpIter performs the operation with the guide of iterators
func (e E) OpIter(t reflect.Type, a, b Array, ait, bit Iterator) error { return nil }

// OpIncr performs the operation and adds the result to the incr Array
func (e E) OpIncr(t reflect.Type, a, b, incr Array) error { return nil }

// OpIterIncr performs the operation and adds the result to the incr Array with the guide ofiterators
func (e E) OpIterIncr(t reflect.Type, a, b, incr Array, ait, bit, iit Iterator) error { return nil }

// Cmp performs comparison operations. It's different in the sense that a retVal is passed in.
func (e E) Cmp(t reflect.Type, same bool, a, b, retVal Array) error { return nil }

func (e E) CmpIter(t reflect.Type, same bool, a, b, retVal Array, ait, bit Iterator) error { return nil }
