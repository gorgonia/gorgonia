package gorgonia

import (
	"fmt"

	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// TypeOf returns the Type of the value
func TypeOf(v Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return makeTensorType(dim, dt)
	case Scalar:
		return t.Dtype()
	case Typer:
		return t.Type()

	default:
		panic(fmt.Sprintf("TypeOf Not yet implemented for %v %T", v, v))
	}
}

func typeCheckTypeOf(v Value) hm.Type {
	switch t := v.(type) {
	case tensor.Tensor:
		dt, dim := tensorInfo(t)
		return newTensorType(dim, dt)
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

// CloneValue clones a value. For scalars, since Go copies scalars, it returns itself
func CloneValue(v Value) (Value, error) {
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

func setEngine(v Value, e tensor.Engine) {
	switch vv := v.(type) {
	case *dualValue:
		setEngine(vv.Value, e)
		setEngine(vv.d, e)
	case tensor.Tensor:
		tensor.WithEngine(e)(vv)
	}
}
