package onnx

import (
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
)

//START_DOC OMIT

// Graph ...
type Graph struct {
	*engine.ExprGraph
}

// ONNXGetOperationFromName ...
func (g Graph) ONNXGetOperationFromName(s string) (interface{}, error) {
	switch s {
	case "Abs":
		return &Abs{}, nil
	case "Sign":
		return &Sign{}, nil
	case "Ceil":
		return &Ceil{}, nil
	case "Floor":
		return &Floor{}, nil
	case "Sin":
		return &Sin{}, nil
	case "Cos":
		return &Cos{}, nil
	case "Exp":
		return &Exp{}, nil
	case "Log":
		return &Log{}, nil
	case "Log2":
		return &Log2{}, nil
	case "Neg":
		return &Neg{}, nil
	case "Square":
		return &Square{}, nil
	case "Sqrt":
		return &Sqrt{}, nil
	case "Inverse":
		return &Inverse{}, nil
	case "InverseSqrt":
		return &InverseSqrt{}, nil
	case "Cube":
		return &Cube{}, nil
	case "Tanh":
		return &Tanh{}, nil
	case "Sigmoid":
		return &Sigmoid{}, nil
	case "Log1p":
		return &Log1p{}, nil
	case "Expm1":
		return &Expm1{}, nil
	case "Softplus":
		return &Softplus{}, nil
	case "Add":
		return &Add{}, nil
	case "Sub":
		return &Sub{}, nil
	case "HadamardProd":
		return &HadamardProd{}, nil
	case "HadamardDiv":
		return &HadamardDiv{}, nil
	case "Pow":
		return &Pow{}, nil
	case "Lt":
		return &Lt{}, nil
	case "Gt":
		return &Gt{}, nil
	case "Lte":
		return &Lte{}, nil
	case "Gte":
		return &Gte{}, nil
	case "Eq":
		return &Eq{}, nil
	case "Ne":
		return &Ne{}, nil
	default:
		return nil, &ErrNotImplemented{
			Operator: s,
		}
	}
}

// ONNXApply ...
func (g Graph) ONNXApply(operation func(g graph.WeightedDirected, n graph.Node) (interface{}, error), n graph.Node) error { // HL
	oper := func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		output, err := operation(g, n)
		return output.(ops.Op), err
	}
	return g.ApplyOp(engine.Operation(oper), n.(*engine.Node))
}

// END_DOC OMIT
