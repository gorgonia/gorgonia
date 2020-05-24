package values

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	gerrors "gorgonia.org/gorgonia/internal/errors"
	"gorgonia.org/gorgonia/internal/memutils"
	"gorgonia.org/gorgonia/types"
	"gorgonia.org/tensor"
)

// AnyToScalar converts any primitive type into a scalar type, and the dtype.
func AnyToScalar(any interface{}) (Scalar, tensor.Dtype) {
	switch at := any.(type) {
	case Scalar:
		return at, at.Dtype()
	case float64:
		return NewF64(at), tensor.Float64
	case float32:
		return NewF32(at), tensor.Float32
	case int:
		return NewI(at), tensor.Int
	case int32:
		return NewI32(at), tensor.Int32
	case int64:
		return NewI64(at), tensor.Int64
	case byte:
		return NewU8(at), tensor.Byte
	case bool:
		return NewB(at), tensor.Bool
	default:
		panic(fmt.Sprintf("%v(%T) not scalar/not handled", any, any))
	}
}

// AnyToValue converts any known type into a Value. It also returns the Type and Dtype.
func AnyToValue(any interface{}) (val Value, t hm.Type, dt tensor.Dtype, err error) {
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
	case F64:
		return NewF64(float64(a)), tensor.Float64, tensor.Float64, nil
	case F32:
		return NewF32(float32(a)), tensor.Float32, tensor.Float32, nil
	case I:
		return NewI(int(a)), tensor.Int, tensor.Int, nil
	case I64:
		return NewI64(int64(a)), tensor.Int64, tensor.Int64, nil
	case I32:
		return NewI32(int32(a)), tensor.Int32, tensor.Int32, nil
	case U8:
		return NewU8(byte(a)), tensor.Uint8, tensor.Uint8, nil
	case B:
		return NewB(bool(a)), tensor.Bool, tensor.Bool, nil
	case tensor.Tensor:
		val = a
		t = TypeOf(a)
		dt = a.Dtype()
		return
	default:
		err = errors.Errorf("value %v of %T not yet handled", any, any)
		return
	}
}

// One creates a Value of the given Dtype with the equivalent value of 1.
func One(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return NewF64(float64(1))
	case tensor.Float32:
		return NewF32(float32(1))
	case tensor.Int:
		return NewI(1)
	case tensor.Int32:
		return NewI32(int32(1))
	case tensor.Int64:
		return NewI64(int64(1))
	case tensor.Byte:
		return NewU8(byte(1))
	case tensor.Bool:
		return NewB(true)
	default:
		panic("Unhandled dtype")
	}
}

// Zero creates a Value of the given Dtype with the equivalent value of 0.
func Zero(dt tensor.Dtype) Scalar {
	switch dt {
	case tensor.Float64:
		return NewF64(float64(0))
	case tensor.Float32:
		return NewF32(float32(0))
	case tensor.Int:
		return NewI(0)
	case tensor.Int32:
		return NewI32(int32(0))
	case tensor.Int64:
		return NewI64(int64(0))
	case tensor.Byte:
		return NewU8(byte(0))
	case tensor.Bool:
		return NewB(false)
	default:
		panic("Unhandled dtype")
	}
}

func MakeFromMem(t hm.Type, s tensor.Shape, mem tensor.Memory) (retVal Value, err error) {
	var dt tensor.Dtype
	if dt, err = types.DtypeOf(t); err != nil {
		return
	}
	if s.IsScalar() {
		return makeScalarFromMem(dt, mem)
	}

	switch tt := t.(type) {
	case types.TensorType:
		memsize := memutils.MemSize(dt, s)
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...), tensor.FromMemory(mem.Uintptr(), uintptr(memsize))), nil
	case tensor.Dtype:
		return makeScalarFromMem(tt, mem)
	default:
		err = errors.Errorf(gerrors.NYITypeFail, "MakeValue", tt)
		return
	}
}

func Make(t hm.Type, s tensor.Shape) (retVal Value, err error) {
	var dt tensor.Dtype
	if dt, err = types.DtypeOf(t); err != nil {
		return
	}

	if s.IsScalar() {
		switch dt {
		case tensor.Float64:
			return NewF64(0), nil
		case tensor.Float32:
			return NewF32(0), nil
		case tensor.Int:
			return NewI(0), nil
		case tensor.Int64:
			return NewI64(0), nil
		case tensor.Int32:
			return NewI32(0), nil
		case tensor.Byte:
			return NewU8(0), nil
		case tensor.Bool:
			return NewB(false), nil
		}
	}

	switch tt := t.(type) {
	case types.TensorType:
		return tensor.New(tensor.Of(dt), tensor.WithShape(s...)), nil
	default:
		err = errors.Errorf(gerrors.NYITypeFail, "MakeValue", tt)
		return
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
	case Scalar:
		if bt, ok := b.(Scalar); ok {
			return scalarEq(at, bt)
		}
		return false
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
	case Scalar:
		if bt, ok := b.(Scalar); ok {
			return scalarClose(at, bt)
		}
		return false
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
	case *F64:
		retVal := *vt
		return &retVal, nil
	case *F32:
		retVal := *vt
		return &retVal, nil
	case *I:
		retVal := *vt
		return &retVal, nil
	case *I32:
		retVal := *vt
		return &retVal, nil
	case *I64:
		retVal := *vt
		return &retVal, nil
	case *U8:
		retVal := *vt
		return &retVal, nil
	case *B:
		retVal := *vt
		return &retVal, nil
	case tensor.Tensor:
		return vt.Clone().(*tensor.Dense), nil
	case CloneErrorer:
		ret, err := vt.Clone()
		if err != nil {
			return nil, err
		}
		retVal, ok := ret.(Value)
		if !ok {
			return nil, errors.Errorf("Cloner is not a value: %v %T", v, v)
		}
		return retVal, nil
	case Cloner:
		return vt.Clone().(Value), nil
	default:
		return nil, errors.Errorf("Unable to clone value of type %T", v)
	}
}

// ZeroValue returns the zero value of a type
func ZeroValue(v Value) Value {
	switch vt := v.(type) {
	case *F64:
		*vt = 0
		return vt
	case *F32:
		*vt = 0
		return vt
	case *I:
		*vt = 0
		return vt
	case *I32:
		*vt = 0
		return vt
	case *I64:
		*vt = 0
		return vt
	case *U8:
		*vt = 0
		return vt
	case *B:
		*vt = false
		return vt
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
	switch srcT := src.(type) {
	case *F64:
		var destS *F64
		if destS, ok = dest.(*F64); !ok {
			return nil, errors.Errorf("Expected dest to be *F64. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *F32:
		var destS *F32
		if destS, ok = dest.(*F32); !ok {
			return nil, errors.Errorf("Expected dest to be *F32. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *I:
		var destS *I
		if destS, ok = dest.(*I); !ok {
			return nil, errors.Errorf("Expected dest to be *I) . Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *I64:
		var destS *I64
		if destS, ok = dest.(*I64); !ok {
			return nil, errors.Errorf("Expected dest to be *I64. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *I32:
		var destS *I32
		if destS, ok = dest.(*I32); !ok {
			return nil, errors.Errorf("Expected dest to be *I32. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *U8:
		var destS *U8
		if destS, ok = dest.(*U8); !ok {
			return nil, errors.Errorf("Expected dest to be *U8). Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *B:
		var destS *B
		if destS, ok = dest.(*B); !ok {
			return nil, errors.Errorf("Expected dest to be *B) . Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case tensor.Tensor:
		var destT tensor.Tensor
		if destT, ok = dest.(tensor.Tensor); !ok {
			return nil, errors.Errorf("Expected dest to be a tensor.Tensor. Got %T instead", dest)
		}
		err := tensor.Copy(destT, srcT)
		return dest, err
	case CopierTo:
		err := srcT.CopyTo(dest)
		return dest, err
	default:
		var copyFrom CopierFrom
		if copyFrom, ok = dest.(CopierFrom); ok {
			err := copyFrom.CopyFrom(src)
			return dest, err
		}
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
