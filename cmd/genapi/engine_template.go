package main

const engineHeader = `package engine 

import (
	"github.com/pkg/errors"
	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/node"
	"gorgonia.org/gorgonia/ops"
)

%v

`
const engineUnaryTemplateRaw = ` // {{.FnName}} performs a pointwise {{lower .FnName}}.
func {{.FnName}}(a *Node) (*Node, error) { return unaryOpNode(newElemUnaryOp({{.OpType}}, a), a) }


// {{.FnName}}Op ...
func New{{.FnName}}Operation() Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		it := getOrderedChildren(g, n)
		if it.Len() != 1 {
			return nil, errors.New("Unexpected number of children")
		}
		children := make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}

		return newElemUnaryOp({{.OpType}}, children[0]), nil
	}
}

`

const engineBinaryTemplateRaw = `// {{.FnName}} perfors a pointwise {{lower .FnName}} operation.
{{if .AsSame -}}// retSame indicates if the data type of the return value should be the same as the input data type. It defaults to Bool otherwise.
{{end -}}
func {{.FnName}}(a, b *Node{{if .AsSame}}, retSame bool{{end}}) (*Node, error) { {{if not .AsSame -}}return binOpNode(newElemBinOp({{.OpType}}, a, b), a, b) {{else -}}
	op := newElemBinOp({{.OpType}}, a, b)
	op.retSame = retSame
	return binOpNode(op, a, b)
{{end -}}
}

// {{.FnName}}Op ...
func New{{.FnName}}Operation(leftAxes, rightAxes []byte{{if .AsSame}}, retSame bool{{end}}) Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		it := getOrderedChildren(g, n)
		if it.Len() != 2 {
			return nil, errors.New("Unexpected number of children")
		}
		children := make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}
		{{if not .AsSame -}}x := children[0]
		y := children[1]

		if leftAxes != nil || rightAxes != nil {
			builder, ok := g.(graph.DirectedWeightedBuilder)
			if !ok {
				return nil, errors.Errorf("Broadcast needs to modify the graph but is not a DirectedWeightedBuilder")
			}
			_, ok = g.(graph.EdgeRemover)
			if !ok {
				return nil, errors.Errorf("Broadcast needs to modify the graph but is not an EdgeRemover")
			}

			pattern := newBroadcastPattern(leftAxes, rightAxes)
			broadcastOn := pattern.on()
			switch {
			case len(broadcastOn[0]) != 0:
				// Remove the link from n to x
				g.(graph.EdgeRemover).RemoveEdge(n.ID(), x.ID())
				broadcastedX := builder.NewNode().(*Node)
				broadcastedX.name = n.(*Node).name + "_broadcastedX"
				builder.AddNode(broadcastedX)
				// Link it to the input tensor
				builder.SetWeightedEdge(builder.NewWeightedEdge(n, broadcastedX, 0.0))
				builder.SetWeightedEdge(builder.NewWeightedEdge(broadcastedX, x, 0.0))
				builder.SetWeightedEdge(builder.NewWeightedEdge(broadcastedX, y, 1.0))

				bcastOp := newBroadcastOperation(second, broadcastOn[0])
				err := g.(*ExprGraph).ApplyOp(bcastOp, broadcastedX)
				if err != nil {
					return nil, err
				}
				//x = broadcastedX
			case len(broadcastOn[1]) != 0:
				// Remove the link from n to x
				g.(graph.EdgeRemover).RemoveEdge(n.ID(), y.ID())
				broadcastedY := builder.NewNode().(*Node)
				broadcastedY.name = n.(*Node).name + "_broadcastedY"
				builder.AddNode(broadcastedY)
				// Link it to the input tensor
				builder.SetWeightedEdge(builder.NewWeightedEdge(n, broadcastedY, 0.0))
				builder.SetWeightedEdge(builder.NewWeightedEdge(broadcastedY, x, 0.0))
				builder.SetWeightedEdge(builder.NewWeightedEdge(broadcastedY, y, 1.0))

				bcastOp := newBroadcastOperation(first, broadcastOn[1])
				err := g.(*ExprGraph).ApplyOp(bcastOp, broadcastedY)
				if err != nil {
					return nil, err
				}
				//y = broadcastedY
			}
		}
		it = getOrderedChildren(g, n)
		if it.Len() != 2 {
			return nil, errors.New("AddOperation: Unexpected number of children")
		}
		children = make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}
		return newElemBinOp({{.OpType}}, children[0], children[1]), nil {{else -}}
		op:= newElemBinOp({{.OpType}}, children[0], children[1])
		op.retSame = retSame
		return op,nil
		{{end -}}
	}
}
`
