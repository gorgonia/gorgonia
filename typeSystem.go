package gorgonia

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
)

// inferType infers the type of the expression
func inferType(expr interface{}) (retVal hm.Type, err error) {
	switch e := expr.(type) {
	case *Node:
		if e.isInput() || e.isConstant() {
			// Var (and Let const)
			return e.t, nil
		}

		// stop the recursive inference early - if the node already has a type, return it
		if e.t != nil {
			return e.t, nil
		}

		return inferNodeType(e.op, e.children...)
	case Op:
		return e.Type(), nil
	case float32:
		return Float32, nil
	case float64:
		return Float64, nil
	case int:
		return Int, nil
	case int64:
		return Int64, nil
	case int32:
		return Int32, nil
	case bool:
		return Bool, nil
	default:
		err = errors.Errorf(nyiTypeFail, "inferType", expr)
		return
	}
}

// Instead of using hm's Infer function, since all the nodes are pretty much hm.Apply, we write our own.
func inferNodeType(op Op, children ...*Node) (retVal hm.Type, err error) {
	fnType := op.Type()
	if fnt, ok := fnType.(*hm.FunctionType); ok {
		defer hm.ReturnFnType(fnt)
	}

	argTypes := hm.BorrowTypes(len(children) + 1)
	defer hm.ReturnTypes(argTypes)
	for i, child := range children {
		if argTypes[i], err = inferType(child); err != nil {
			return nil, errors.Wrapf(err, "Failed to infer type of %v", child)
		}
	}

	b := hm.TypeVariable('b')
	argTypes[len(argTypes)-1] = b

	fn := hm.NewFnType(argTypes...)
	defer hm.ReturnFnType(fn)

	// var t0 hm.Type
	var sub hm.Subs
	if sub, err = hm.Unify(fn, fnType); err != nil {
		return nil, errors.Wrapf(err, "Unable to unify while inferring type of %v", op)
	}

	var ok bool
	if retVal, ok = sub.Get(b); !ok {
		return nil, errors.Errorf("Expected a replacement for %v", b)
	}

	// return pruneReturn(t0.(*hm.FunctionType).ReturnType()), nil
	return retVal, nil
}

func isScalarType(t hm.Type) bool {
	switch tt := t.(type) {
	case tensor.Dtype:
		return true
	case TensorType:
		if tt.Dims == 0 {
			return true
		}
		return false
	case hm.TypeVariable:
		panic("Type Variable is a type that is not yet known.")
	default:
		panic("Unhandled type")
	}
}

func dtypeOf(t hm.Type) (retVal tensor.Dtype, err error) {
	switch p := t.(type) {
	case tensor.Dtype:
		retVal = p
	case TensorType:
		return dtypeOf(p.Of)
	case hm.TypeVariable:
		err = errors.Errorf("instance %v does not have a dtype", p)
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
