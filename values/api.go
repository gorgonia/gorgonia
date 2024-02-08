package values

import (
	"fmt"

	"github.com/chewxy/hm"

	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/internal/datatypes"
	"gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/shapes"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/dense"
	"gorgonia.org/tensor/scalar"
)

// AnyToScalar converts any primitive type into a scalar type, and the dtype.
func AnyToScalar[DT any](x DT) (Value[DT], dtype.Dtype) {
	retVal := dense.New[DT](tensor.WithShape(), tensor.WithBacking([]DT{x}))
	return retVal, retVal.Dtype()
}

// AnyToValue converts any known type into a Value. It also returns the Type and Dtype.
func AnyToValue[DT any](any interface{}) (val Value[DT], t hm.Type, dt dtype.Dtype, err error) {
	panic("NYI")
	// switch a := any.(type) {
	// case Value:
	// 	val = a
	// 	t = TypeOf(a)
	// 	dt = a.Dtype()
	// 	return
	// case float64, float32, int, int64, int32, byte, bool:
	// 	val, dt = AnyToScalar(any)
	// 	t = dt
	// 	return
	// default:
	// 	err = errors.Errorf("value %v of %T not yet handled", any, any)
	// 	return
	// }
}

// One creates a scalar of the given Dtype with the equivalent value of 1.
func One[DT any]() DT {
	var z DT
	switch o := any(z).(type) {
	case uint8:
		return any(o + 1).(DT)
	case uint16:
		return any(o + 1).(DT)
	case uint32:
		return any(o + 1).(DT)
	case uint64:
		return any(o + 1).(DT)
	case uint:
		return any(o + 1).(DT)
	case int8:
		return any(o + 1).(DT)
	case int16:
		return any(o + 1).(DT)
	case int32:
		return any(o + 1).(DT)
	case int64:
		return any(o + 1).(DT)
	case float32:
		return any(o + 1).(DT)
	case float64:
		return any(o + 1).(DT)
	case complex64:
		return any(o + 1).(DT)
	case complex128:
		return any(o + 1).(DT)
	case bool:
		return any(true).(DT)
	case ScalarOner[DT]:
		return o.ScalarOne()
	default:
		panic(fmt.Sprintf("Cannot get One for %T", z))
	}
}

// nativeOne generates the given "1" value and returns it as an interface.
// The reason for abstracting out this function is because this function is
// linkname'd in various other subpackages.
func nativeOne(dt dtype.Dtype) interface{} {
	r, err := dtype.FromInt(dt, 1)
	if err != nil {
		panic(err)
	}
	return r
}

// Zero creates a Value of the given Dtype with the equivalent value of 0.
func Zero[DT any]() scalar.Scalar[DT] { return scalar.Scalar[DT]{} }

func ZeroV(dt dtype.Dtype) tensor.DescWithStorage {
	return scalar.Z(dt)
}

// nativeZero generates the given "0" value and returns it as an interface.
// The reason for abstracting out this function is because this function is
// linkname'd in various other subpackages.
func nativeZero(dt dtype.Dtype) interface{} {
	panic("NYI")
	// return reflect.Zero(dt.Type).Interface()
}

func MakeFromMem[DT any](t hm.Type, s shapes.Shape, mem tensor.Memory) (retVal Value[DT], err error) {
	panic("NYI")
	// var dt dtype.Dtype
	// if dt, err = datatypes.DtypeOf(t); err != nil {
	// 	return
	// }
	// if s.IsScalar() {
	// 	return makeScalarFromMem(dt, mem)
	// }

	// switch tt := t.(type) {
	// case types.TensorType:
	// 	memsize := memutils.MemSize(dt, s)
	// 	return dense.NewOf(dt, tensor.WithShape(s...), tensor.FromMemory(mem.Uintptr(), uintptr(memsize)))
	// case dtype.Dtype:
	// 	return makeScalarFromMem(tt, mem)
	// default:
	// 	return nil, errors.NYI(tt)
	// }

}

func Make(t hm.Type, s shapes.Shape) (retVal tensor.DescWithStorage, err error) {
	var dt dtype.Dtype
	if dt, err = datatypes.DtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		return ZeroV(dt), nil
	}

	switch tt := t.(type) {
	case types.TensorType:
		return dense.NewOf(dt, tensor.WithShape(s...))
	default:
		return nil, errors.NYI(tt)
	}
}

// TypeOf returns the Type of the value
func TypeOf(v any) hm.Type {
	switch t := v.(type) {
	case tensor.DescWithStorage:
		dt, dim := tensorInfo(t)
		return types.MakeTensorType(dim, dt)
	case Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

// ValueEq is the equality function for values
func ValueEq[DT comparable](a, b Value[DT]) bool {
	if a == nil && b == nil {
		return true
	}
	switch at := a.(type) {
	case ValueEqualer[DT]:
		return at.ValueEq(b)
	case scalar.Scalar[DT]:
		if bt, ok := b.(scalar.Scalar[DT]); ok {
			return at.V == bt.V
		}
		return false
	case *dense.Dense[DT]:
		if bt, ok := b.(*dense.Dense[DT]); ok {
			return at.Eq(bt)
		}
		return false
	}
	panic("Unreachable")

}

// ValueClose checks whether two values are close to one another. It's predominantly used as an alternative equality test for floats
func ValueClose[DT any](a, b Value[DT]) bool {
	panic("NYI")
	// if a == nil && b == nil {
	// 	return true
	// }

	// switch at := a.(type) {
	// case tensor.Basic[DT]:
	// 	if bt, ok := b.(tensor.Basic[DT]); ok {
	// 		return tensorClose(at, bt)
	// 	}
	// 	return false
	// case ValueCloser[DT]:
	// 	return at.ValueClose(b)
	// default:
	// 	panic("Not implemented yet")
	// }
}

// Clone clones a value. For scalars, since Go copies scalars, it returns itself
func Clone[T Cloner[T]](v T) T {
	return v.Clone()
}

// ShallowClone clones a value without making a copy of the underlying data.
// If A and B are shallow clones of C then if A modifies the data, B will see the modification too.
// The whole purpose of ShallowClone is for there to be multiple "views" of the same underlying data:
// e.g. A and B could be reshapes or slices of C.
func ShallowClone[T tensor.ShallowCloner[T]](v T) T {
	return v.ShallowClone()
}

// ZeroValue returns the zero value of a type
func ZeroValue[DT any](v Value[DT]) Value[DT] {
	switch vt := v.(type) {
	case scalar.Scalar[DT]:
		return scalar.Scalar[DT]{}
	case tensor.Basic[DT]:
		vt.Zero()
		return vt
	default:
		panic(fmt.Sprintf("Cannot return zero value of %T", v))
	}
}

// Copy copies the src values into dest values. For scalars, it just returns itself
func Copy[DT any](dest, src Value[DT]) (Value[DT], error) {

	var ok bool
	var copyFrom CopierFrom
	if copyFrom, ok = dest.(CopierFrom); ok {
		err := copyFrom.CopyFrom(src)
		return dest, err
	}

	switch srcT := src.(type) {
	case CopierTo[*dense.Dense[DT]]:
		err := srcT.CopyTo(dest.(*dense.Dense[DT]))
		return dest, err
	default:
		return nil, errors.Errorf("Unable to copy value of type %T into value of type %T", src, dest)
	}
}

// SetEngine sets the engine of the given value.
func SetEngine[DT any](v Value[DT], e tensor.Engine) {
	switch vv := v.(type) {
	case engineSetter:
		vv.SetEngine(e)
	default:

	}
}

type engineSetter interface {
	SetEngine(e tensor.Engine)
}
