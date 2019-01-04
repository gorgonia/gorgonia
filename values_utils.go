package gorgonia

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia/internal/primitive"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

// TypeOf returns the Type of the value
func TypeOf(v value.Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return makeTensorType(dim, dt)
	case value.Scalar:
		return t.Dtype()
	case value.Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

func typeCheckTypeOf(v value.Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return newTensorType(dim, dt)
	case value.Scalar:
		return t.Dtype()
	case value.Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

// ValueEq is the equality function for values
func ValueEq(a, b value.Value) bool {
	if a == nil && b == nil {
		return true
	}
	switch at := a.(type) {
	case value.Scalar:
		if bt, ok := b.(value.Scalar); ok {
			return scalarEq(at, bt)
		}
		return false
	case tensor.Tensor:
		if bt, ok := b.(tensor.Tensor); ok {
			return at.Eq(bt)
		}
		return false
	case value.ValueEqualer:
		return at.ValueEq(b)
	default:
		panic(fmt.Sprintf("Not implemented yet, %T", a))
	}
}

// ValueClose checks whether two values are close to one another. It's predominantly used as an alternative equality test for floats
func ValueClose(a, b value.Value) bool {
	if a == nil && b == nil {
		return true
	}

	switch at := a.(type) {
	case value.Scalar:
		if bt, ok := b.(value.Scalar); ok {
			return scalarClose(at, bt)
		}
		return false
	case tensor.Tensor:
		if bt, ok := b.(tensor.Tensor); ok {
			return tensorClose(at, bt)
		}
		return false
	case value.ValueCloser:
		return at.ValueClose(b)
	default:
		panic("Not implemented yet")
	}
}

// CloneValue clones a value. For scalars, since Go copies scalars, it returns itself
func CloneValue(v value.Value) (value.Value, error) {
	switch vt := v.(type) {
	case *primitive.F64:
		retVal := *vt
		return &retVal, nil
	case *primitive.F32:
		retVal := *vt
		return &retVal, nil
	case *primitive.I:
		retVal := *vt
		return &retVal, nil
	case *primitive.I32:
		retVal := *vt
		return &retVal, nil
	case *primitive.I64:
		retVal := *vt
		return &retVal, nil
	case *primitive.U8:
		retVal := *vt
		return &retVal, nil
	case *primitive.B:
		retVal := *vt
		return &retVal, nil
	case tensor.Tensor:
		return vt.Clone().(*tensor.Dense), nil
	case value.CloneErrorer:
		ret, err := vt.Clone()
		if err != nil {
			return nil, err
		}
		retVal, ok := ret.(value.Value)
		if !ok {
			return nil, errors.Errorf("Cloner is not a value: %v %T", v, v)
		}
		return retVal, nil
	case value.Cloner:
		return vt.Clone().(value.Value), nil
	default:
		return nil, errors.Errorf("Unable to clone value of type %T", v)
	}
}

// ZeroValue returns the zero value of a type
func ZeroValue(v value.Value) value.Value {
	switch vt := v.(type) {
	case *primitive.F64:
		*vt = 0
		return vt
	case *primitive.F32:
		*vt = 0
		return vt
	case *primitive.I:
		*vt = 0
		return vt
	case *primitive.I32:
		*vt = 0
		return vt
	case *primitive.I64:
		*vt = 0
		return vt
	case *primitive.U8:
		*vt = 0
		return vt
	case *primitive.B:
		*vt = false
		return vt
	case tensor.Tensor:
		vt.Zero()
		return vt
	case value.ZeroValuer:
		return vt.ZeroValue()
	default:
		panic(fmt.Sprintf("Cannot return zero value of %T", v))
	}
}

// Copy copies the src values into dest values. For scalars, it just returns itself
func Copy(dest, src value.Value) (value.Value, error) {
	var ok bool
	switch srcT := src.(type) {
	case *primitive.F64:
		var destS *primitive.F64
		if destS, ok = dest.(*primitive.F64); !ok {
			return nil, errors.Errorf("Expected dest to be *primitive.F64. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.F32:
		var destS *primitive.F32
		if destS, ok = dest.(*primitive.F32); !ok {
			return nil, errors.Errorf("Expected dest to be *primitive.F32. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.I:
		var destS *primitive.I
		if destS, ok = dest.(*primitive.I); !ok {
			return nil, errors.Errorf("Expected dest to be *I) . Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.I64:
		var destS *primitive.I64
		if destS, ok = dest.(*primitive.I64); !ok {
			return nil, errors.Errorf("Expected dest to be *I64. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.I32:
		var destS *primitive.I32
		if destS, ok = dest.(*primitive.I32); !ok {
			return nil, errors.Errorf("Expected dest to be *I32. Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.U8:
		var destS *primitive.U8
		if destS, ok = dest.(*primitive.U8); !ok {
			return nil, errors.Errorf("Expected dest to be *U8). Got %T instead", dest)
		}
		*destS = *srcT
		return destS, nil
	case *primitive.B:
		var destS *primitive.B
		if destS, ok = dest.(*primitive.B); !ok {
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
	case value.CopierTo:
		err := srcT.CopyTo(dest)
		return dest, err
	default:
		var copyFrom value.CopierFrom
		if copyFrom, ok = dest.(value.CopierFrom); ok {
			err := copyFrom.CopyFrom(src)
			return dest, err
		}
		return nil, errors.Errorf("Unable to copy value of type %T into value of type %T", src, dest)
	}
}

func setEngine(v value.Value, e tensor.Engine) {
	switch vv := v.(type) {
	case *dualValue:
		setEngine(vv.Value, e)
		setEngine(vv.d, e)
	case tensor.Tensor:
		tensor.WithEngine(e)(vv)
	}
}
