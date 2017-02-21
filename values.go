package gorgonia

import (
	"fmt"
	"unsafe"

	"github.com/chewxy/gorgonia/tensor"
)

// Value represents a value that Gorgonia accepts
type Value interface {
	Shape() tensor.Shape
	Size() int
	Data() interface{}

	fmt.Formatter
}

// Memory is a representation of memory of the value
type Memory interface {
	Uintptr() uintptr
	MemSize() uintptr
	Pointer() unsafe.Pointer
}

type Valuer interface {
	Value() Value
}

type Zeroer interface {
	Value
	Zero()
}

type ZeroValuer interface {
	Value
	ZeroValue() Value
}

type Setter interface {
	SetAll(interface{}) error
}
