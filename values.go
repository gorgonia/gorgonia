package gorgonia

import (
	"fmt"

	"github.com/chewxy/gorgonia/tensor/types"
)

// Value represents a value that Gorgonia accepts
type Value interface {
	Shape() types.Shape
	Size() int
	Data() interface{}

	fmt.Formatter
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
