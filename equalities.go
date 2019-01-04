package gorgonia

import (
	"gorgonia.org/dawson"
	"gorgonia.org/gorgonia/internal/primitive"
	"gorgonia.org/gorgonia/internal/value"
	"gorgonia.org/tensor"
)

func scalarEq(a, b value.Scalar) bool {
	switch at := a.(type) {
	case *primitive.F64:
		if bt, ok := b.(*primitive.F64); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.F32:
		if bt, ok := b.(*primitive.F32); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.I:
		if bt, ok := b.(*primitive.I); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.I32:
		if bt, ok := b.(*primitive.I32); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.I64:
		if bt, ok := b.(*primitive.I64); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.U8:
		if bt, ok := b.(*primitive.U8); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	case *primitive.B:
		if bt, ok := b.(*primitive.B); ok {
			if at == bt {
				return true
			}
			return *at == *bt
		}
		return false
	}
	return false
}

func scalarClose(a, b value.Scalar) bool {
	switch at := a.(type) {
	case *primitive.F64:
		if bt, ok := b.(*primitive.F64); ok {
			return dawson.CloseF64(float64(*at), float64(*bt))
		}
		return false
	case *primitive.F32:
		if bt, ok := b.(*primitive.F32); ok {
			return dawson.CloseF32(float32(*at), float32(*bt))
		}
		return false
	default:
		return scalarEq(a, b)
	}
}

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

/*
func axesEq(a, b axes) bool {
	if len(a) != len(b) {
		return false
	}

	for i, s := range a {
		if b[i] != s {
			return false
		}
	}
	return true
}

// yes it's exactly the same as axesEq
func coordEq(a, b coordinates) bool {
	if len(a) != len(b) {
		return false
	}

	for i, s := range a {
		if b[i] != s {
			return false
		}
	}
	return true
}
*/

func constEq(a, b constant) (ok bool) {
	switch at := a.(type) {
	case constantScalar:
		var bt constantScalar
		if bt, ok = b.(constantScalar); !ok {
			return
		}

		return bt == at
	case constantTensor:
		var bt constantTensor
		if bt, ok = b.(constantTensor); !ok {
			return
		}
		return at.v.Eq(bt.v)
	default:
		panic("Not yet implemented")
	}
}

// fastest comparisons to least fastest
func nodeEq(a, b *Node) bool {
	if a == b {
		return true
	}

	if a.isInput() {
		if !b.isInput() {
			return false
		}
		return a.name == b.name
	}

	if b.isInput() {
		return false
	}

	// hashcode is good for comparing Op (TODO: benchmark this vs reflect.DeepEq)
	if a.op.Hashcode() != b.op.Hashcode() {
		return false
	}

	if len(a.children) != len(b.children) {
		return false
	}

	if a.t != b.t {
		return false
	}

	if !a.shape.Eq(b.shape) {
		return false
	}

	return true
}
