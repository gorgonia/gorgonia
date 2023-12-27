package datatypes

import (
	"fmt"

	"github.com/chewxy/hm"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/tensor"
)

// Tensor represents values that are acceptable in Gorgonia. At this point, it is implemented by:
//   - *exprgraph.Value[DT,T]
//   - *exprgraph.Symbolic[DT]
//   - *dense.Dense[DT]
//   - *dual.Dual[DT]
//   - scalar.Scalar
//
// There is an overlap with values.Value. The reason is semantic clarity. Values are Tensors. Tensors are Values.
type Tensor interface {
	tensor.Desc
	tensor.DataSizer

	// Flags returns the memory flags of the underlying data array.
	//Flags() tensor.MemoryFlag
	// DataOrder returns the data order of the underlying data array.
	//DataOrder() tensor.DataOrder

	// Some basic operations that does not need knowledge of datatype

	// A basic tensor should be able to reshape itself
	//Reshape(shape ...int) error

	// A basic tensor should be able to unsqueeze itself
	//Unsqueeze(axis int) error

	// A Basic tensor should be able to zero itself out
	//tensor.Zeroer

	// Data access related methods

	//RequiresIterator() bool
	//Iterator() tensor.Iterator
	//IsMaterializable() bool

	// Memory and operation related methods

	//tensor.Memory
	tensor.Engineer
	//IsNativelyAccessible() bool // Can Go access the memory?
	//IsManuallyManaged() bool    // Must Go manage the memory
}

// TypeOf returns the type of a given Tensor
func TypeOf[DT any](t Tensor) hm.Type {
	switch tt := t.(type) {
	case hm.Typer:
		return tt.Type()
	default:
		panic(fmt.Sprintf("%v of %T is currently unsupported", tt, tt))
	}
}

// DtypeOf returns the dtype of a given type.
//
// If the input hm.Type is not a parameterized type, or a Dtype, an error will be returned.
func DtypeOf(t hm.Type) (retVal dtype.Dtype, err error) {
	switch p := t.(type) {
	case dtype.Dtype:
		retVal = p
	case types.TensorType:
		return DtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
	default:
		err = errors.NYI(p)
		return
	}

	return
}
