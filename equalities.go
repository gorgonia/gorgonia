package gorgonia

import "reflect"

func scalarEq(a, b Scalar) bool {
	switch at := a.(type) {
	case *F64:
		if bt, ok := b.(*F64); ok {
			return *at == *bt
		}
		return false
	case *F32:
		if bt, ok := b.(*F32); ok {
			return *at == *bt
		}
		return false
	case *I:
		if bt, ok := b.(*I); ok {
			return *at == *bt
		}
		return false
	case *I32:
		if bt, ok := b.(*I32); ok {
			return *at == *bt
		}
		return false
	case *I64:
		if bt, ok := b.(*I64); ok {
			return *at == *bt
		}
		return false
	case *U8:
		if bt, ok := b.(*U8); ok {
			return *at == *bt
		}
		return false
	case *B:
		if bt, ok := b.(*B); ok {
			return *at == *bt
		}
		return false
	}
	return false
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
		return reflect.DeepEqual(at, bt)
	default:
		panic("Not yet implemented")
	}
	panic("unreachable")
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
