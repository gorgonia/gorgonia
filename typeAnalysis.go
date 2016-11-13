package gorgonia

import "github.com/pkg/errors"

// inferType infers the type of the expression
func inferType(expr interface{}, nonGenerics typeSet) (retVal Type, err error) {
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
	}
	panic("Unreachable infer")
}

func inferNodeType(op Op, children ...*Node) (retVal Type, err error) {
	optype := op.Type()
	if fnt, ok := optype.(*functionType); ok {
		defer returnFnType(fnt)
	}
	typeSysLogf("optype: %v", optype)

	enterLoggingContext()
	defer leaveLoggingContext()

	// nonGenerics := NewTypeSet()
	argTypes := make(Types, len(children)+1)
	for i, child := range children {
		typeSysLogf("child %d %v type: %v;", i, child, child.t)
		if argTypes[i], err = inferType(child, nil); err != nil {
			return nil, errors.Wrap(err, "Failled to carry inferType()")
		}
	}

	retType := newTypeVariable("b")
	argTypes[len(argTypes)-1] = retType

	fnType := newFunctionType(argTypes...)
	defer returnFnType(fnType)

	typeSysLogf("realized fnType: %v; opType: %v", fnType, optype)

	if err = unify(fnType, optype); err != nil {
		return nil, errors.Wrap(err, "Failed to carry unify()")
	}

	retVal = pruneCompletely(retType)
	return
}
