package values

import (
	"fmt"
	"reflect"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/dtype"
	"gorgonia.org/gorgonia/internal/datatypes"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/internal/memutils"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/tensor"
)

// AnyToScalar converts any primitive type into a scalar type, and the dtype.
func AnyToScalar(any interface{}) (Scalar, dtype.Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, at.Dtype()
	case float64:
		return MakeScalar(any), Float64
	case float32:
		return MakeScalar(any), Float32
	case int:
		return MakeScalar(any), Int
	case int32:
		return MakeScalar(any), Int32
	case int64:
		return MakeScalar(any), Int64
	case byte:
		return MakeScalar(any), Byte
	case bool:
		return MakeScalar(any), Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled", any, any))
	}
}

// AnyToValue converts any known type into a Value. It also returns the Type and Dtype.
func AnyToValue(any interface{}) (val Value, t hm.Type, dt dtype.Dtype, err error) {
	switch a := any.(type) {
	case Value:
		val = a
		t = TypeOf(a)
		dt = a.Dtype()
		return
	case float64, float32, int, int64, int32, byte, bool:
		val, dt = AnyToScalar(any)
		t = dt
		return
	default:
		err = errors.Errorf("value %v of %T not yet handled", any, any)
		return
	}
}

// One creates a Value of the given Dtype with the equivalent value of 1.
func One(dt dtype.Dtype) Scalar {
	return MakeScalar(nativeOne(dt))
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
func Zero(dt dtype.Dtype) Scalar { return MakeScalar(nativeZero(dt)) }

// nativeZero generates the given "0" value and returns it as an interface.
// The reason for abstracting out this function is because this function is
// linkname'd in various other subpackages.
func nativeZero(dt dtype.Dtype) interface{} { return reflect.Zero(dt.Type).Interface() }

func MakeFromMem(t hm.Type, s tensor.Shape, mem tensor.Memory) (retVal Value, err error) {
	var dt dtype.Dtype
	if dt, err = datatypes.DtypeOf(t); err != nil {
		return
	}
	if s.IsScalar() {
		return makeScalarFromMem(dt, mem)
	}

	switch tt := t.(type) {
	case types.TensorType:
		memsize := memutils.MemSize(dt, s)
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...), tensor.FromMemory(mem.Uintptr(), uintptr(memsize))), nil
	case dtype.Dtype:
		return makeScalarFromMem(tt, mem)
	default:
		return nil, gerrors.NYI(tt)
	}
}

func Make(t hm.Type, s tensor.Shape) (retVal Value, err error) {
	var dt dtype.Dtype
	if dt, err = datatypes.DtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		return Zero(dt), nil
	}

	switch tt := t.(type) {
	case types.TensorType:
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...)), nil
	default:
		return nil, gerrors.NYI(tt)
	}
}

// TypeOf returns the Type of the value
func TypeOf(v Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return types.MakeTensorType(dim, dt)
	case Scalar:
		return t.Dtype()
	case Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

// ValueEq is the equality function for values
func ValueEq(a, b Value) bool {
	if a == nil && b == nil {
		return true
	}
	switch at := a.(type) {
	case tensor.Tensor:
		if bt, ok := b.(tensor.Tensor); ok {
			return at.Eq(bt)
		}
		return false
	case ValueEqualer:
		return at.ValueEq(b)
	default:
		panic(fmt.Sprintf("Not implemented yet, %T", a))
	}
}

// ValueClose checks whether two values are close to one another. It's predominantly used as an alternative equality test for floats
func ValueClose(a, b Value) bool {
	if a == nil && b == nil {
		return true
	}

	switch at := a.(type) {
	case tensor.Tensor:
		if bt, ok := b.(tensor.Tensor); ok {
			return tensorClose(at, bt)
		}
		return false
	case ValueCloser:
		return at.ValueClose(b)
	default:
		panic("Not implemented yet")
	}
}

// Clone clones a value. For scalars, since Go copies scalars, it returns itself
func Clone(v Value) (Value, error) {
	switch vt := v.(type) {
	case Scalar:
		return vt.Clone().(Scalar), nil
	case tensor.Tensor:
		return vt.Clone().(*tensor.Dense), nil
	case Cloner:
		return vt.Clone().(Value), nil
	default:
		return nil, errors.Errorf("Unable to clone value of type %T", v)
	}
}

// ShallowClone clones a value without making a copy of the underlying data.
// If A and B are shallow clones of C then if A modifies the data, B will see the modification too.
// The whole purpose of ShallowClone is for there to be multiple "views" of the same underlying data:
// e.g. A and B could be reshapes or slices of C.
func ShallowClone(v Value) (Value, error) {
	switch vt := v.(type) {
	case denseShallowCloner:
		return vt.ShallowClone(), nil
	default:
		return nil, errors.Errorf("Unable to shallow clone value of type %T", v)
	}
}

// ZeroValue returns the zero value of a type
func ZeroValue(v Value) Value {
	switch vt := v.(type) {
	case tensor.Tensor:
		vt.Zero()
		return vt
	case ZeroValuer:
		return vt.ZeroValue()
	default:
		panic(fmt.Sprintf("Cannot return zero value of %T", v))
	}
}

// Copy copies the src values into dest values. For scalars, it just returns itself
func Copy(dest, src Value) (Value, error) {
	var ok bool

	var copyFrom CopierFrom
	if copyFrom, ok = dest.(CopierFrom); ok {
		err := copyFrom.CopyFrom(src)
		return dest, err
	}

	switch srcT := src.(type) {
	case CopierTo:
		err := srcT.CopyTo(dest)
		return dest, err
	case tensor.Tensor:
		var destT tensor.Tensor
		if destT, ok = dest.(tensor.Tensor); !ok {
			return nil, errors.Errorf("Expected dest to be a tensor.Tensor. Got %T instead", dest)
		}
		err := tensor.Copy(destT, srcT)
		return dest, err

	default:
		return nil, errors.Errorf("Unable to copy value of type %T into value of type %T", src, dest)
	}
}

// SetEngine sets the engine of the given value.
func SetEngine(v Value, e tensor.Engine) {
	switch vv := v.(type) {
	case tensor.Tensor:
		tensor.WithEngine(e)(vv)
	case engineSetter:
		vv.SetEngine(e)
	}
}

type engineSetter interface {
	SetEngine(e tensor.Engine)
}

type denseShallowCloner interface {
	ShallowClone() *tensor.Dense
}
