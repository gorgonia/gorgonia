package gorgonia

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

func isScalarType(t hm.Type) bool {
	switch tt := t.(type) {
	case Dtype:
		return true
	case TensorType:
		if tt.d == 0 {
			return true
		}
		return false
	case hm.TypeVariable:
		if tt.Instance() == nil {
			panic("Undefined Instance")
		}

		return isScalarType(hm.Prune(tt))
	default:
		panic("Unhandled type")
	}
}

func dtypeOf(t hm.Type) (retVal Dtype, err error) {
	pruned := hm.Prune(t)
	switch p := pruned.(type) {
	case Dtype:
		retVal = p
	case TensorType:
		return dtypeOf(p.of)
	case hm.TypeVariable:
		if p.Instance() == nil {
			err = errors.Errorf("instance %v does not have a dtype", p)
			return
		}

		return dtypeOf(p.Instance())
	default:
		err = errors.Errorf(nyiFail, "dtypeOf", p)
		return
	}

	return

}

// DEPRECATED

/*
func runtimeTypeCheck(expected, got hm.Types) (of Dtype, err error) {
	if len(expected) != len(got) {
		err = NewError(RuntimeError, "Input length mismatch")
		return
	}

	if of, err = dtypeOf(expected[0]); err != nil {
		return
	}

	for i, e := range expected {
		g := got[i]
		if !e.Eq(g) {
			err = NewError(RuntimeError, "Expected input[%d] to be %v. Got %v instead", i, e, got[i])
			return
		}

		if i > 0 {
			var gdt Dtype
			if gdt, err = dtypeOf(g); err == nil {
				if gdt != of {
					err = NewError(RuntimeError, "Different dtypes encountered... Expected %v. Got %v instead", of, gdt)
					return
				}
			} else {
				return
			}
		}
	}
	return
}
*/
