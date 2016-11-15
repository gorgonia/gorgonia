package gorgonia

import (
	"github.com/chewxy/hm"
	"github.com/pkg/errors"
)

// inferType infers the type of the expression
func inferType(expr interface{}, nonGenerics hm.Types) (retVal hm.Type, err error) {
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

func inferNodeType(op Op, children ...*Node) (retVal hm.Type, err error) {
	fnType := op.Type()
	argTypes := make(hm.Types, len(children)+1)
	for i, child := range children {
		if argTypes[i], err = inferType(child, nil); err != nil {
			return nil, errors.Wrap(err, "Failled to carry inferType()")
		}
	}
	argTypes[len(argTypes)-1] = hm.NewTypeVar("b")

	fn := hm.NewFnType(argTypes...)

	var t0 hm.Type
	var r map[hm.TypeVariable]hm.Type
	if t0, _, r, err = hm.Unify(fn, fnType); err != nil {
		return nil, errors.Wrap(err, "Unable to unify")
	}
	retVal = t0.(*hm.FunctionType).ReturnType()

	logf("argtypes %v", argTypes)
	logf("t0 %v", t0)
	logf("r: %v", r)
	logf("retVal %v", retVal)
	return

}
