package gorgonia

import "github.com/chewxy/gorgonia/tensor/types"

// Value represents a value that Gorgonia accepts
type Value interface {
	Shape() types.Shape
	Size() int
	Data() interface{}
}

type Valuer interface {
	Value() Value
}

type Zeroer interface {
	Value
	Zero()
}

type Setter interface {
	SetAll(interface{}) error
}
