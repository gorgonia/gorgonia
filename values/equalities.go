package values

import (
	"gorgonia.org/dawson"
	"gorgonia.org/tensor"
)

func scalarEq(a, b Scalar) bool    { return true }
func scalarClose(a, b Scalar) bool { return true }

/*
func scalarEq(a, b Scalar) bool {
	switch at := a.(type) {
	case *F64:
		if bt, ok := b.(*F64); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *F32:
		if bt, ok := b.(*F32); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *I:
		if bt, ok := b.(*I); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *I32:
		if bt, ok := b.(*I32); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *I64:
		if bt, ok := b.(*I64); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *U8:
		if bt, ok := b.(*U8); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *B:
		if bt, ok := b.(*B); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	}
	return false
}

func scalarClose(a, b Scalar) bool {
	switch at := a.(type) {
	case *F64:
		if bt, ok := b.(*F64); ok {
			return dawson.CloseF64(float64(*at), float64(*bt))
		}
		return false
	case *F32:
		if bt, ok := b.(*F32); ok {
			return dawson.CloseF32(float32(*at), float32(*bt))
		}
		return false
	default:
		return scalarEq(a, b)
	}
}
*/

func tensorClose(a, b tensor.Tensor) bool {
	aDt := a.Dtype()
	bDt := b.Dtype()
	if aDt != bDt {
		return false
	}

	switch aDt {
	case tensor.Float64:
		aFs := a.Data().([]float64)
		bFs := b.Data().([]float64)
		if len(aFs) != len(bFs) {
			return false
		}
		aFs = aFs[:len(aFs)]
		bFs = bFs[:len(aFs)]
		for i, v := range aFs {
			if !dawson.CloseF64(v, bFs[i]) {
				return false
			}
		}
		return true
	case tensor.Float32:
		aFs := a.Data().([]float32)
		bFs := b.Data().([]float32)
		if len(aFs) != len(bFs) {
			return false
		}
		aFs = aFs[:len(aFs)]
		bFs = bFs[:len(aFs)]
		for i, v := range aFs {
			if !dawson.CloseF32(v, bFs[i]) {
				return false
			}
		}
		return true
	default:
		return a.Eq(b)
	}

}
