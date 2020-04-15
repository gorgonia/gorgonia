package gorgonnx

import (
	"fmt"

	"github.com/owulveryck/onnx-go"
)

func register(optype string, op func() operator) {
	operators[optype] = op
}

var operators = map[string]func() operator{}

//var operators = map[string]operator{}

type operator interface {
	// apply analyse the graph to find the children of the node
	// then extract its gorgonia.Node references
	// and assign the result of the operation to the node n
	apply(g *Graph, n *Node) error
	// init the operator with name and attributes as carried by the onnx.Operator
	init(o onnx.Operation) error
}

// check conditions of the children.
// It returns an error is:
//  * children's length != arity
//  * if at least one of the children's pointer fo gorgoniaNode is nil
func checkCondition(children []*Node, arity int) error {
	if len(children) != arity {
		return fmt.Errorf("bad arity for operation (have %v, want %v)", len(children), arity)
	}
	// fail fast
	for i := range children {
		if children[i].gorgoniaNode == nil {
			return fmt.Errorf("at least one of the children node is nil")
		}
	}

	return nil
}
