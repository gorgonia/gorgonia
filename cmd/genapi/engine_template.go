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
func New{{.FnName}}Operation({{if .AsSame}} retSame bool{{end}}) Operation {
	return func(g graph.WeightedDirected, n node.Node) (ops.Op, error) {
		it := getOrderedChildren(g, n)
		if it.Len() != 2 {
			return nil, errors.New("Unexpected number of children")
		}
		children := make([]*Node, it.Len())
		for i := 0; it.Next(); i++ {
			children[i] = it.Node().(*Node)
		}
		{{if not .AsSame -}}return newElemBinOp({{.OpType}}, children[0], children[1]), nil {{else -}}
		op:= newElemBinOp({{.OpType}}, children[0], children[1])
		op.retSame = retSame
		return op,nil
		{{end -}}
	}
}
`
