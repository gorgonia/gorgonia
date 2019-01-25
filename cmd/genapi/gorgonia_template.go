package main

const gorgoniaHeader = `package gorgonia

import (
	"gorgonia.org/gorgonia/internal/engine"
	"gorgonia.org/gorgonia/node"
)

%v

`
const gorgoniaUnaryTemplateRaw = ` // {{.FnName}} performs a pointwise {{lower .FnName}}.
func {{.FnName}}(g *Graph, a node.Node) (node.Node, error) { 
	retval := g.g.NewNode().(*engine.Node)
	g.g.AddNode(retval)
	g.g.SetWeightedEdge(g.g.NewWeightedEdge(retval, a, 1.0))
	err := g.g.ApplyOp(engine.New{{.FnName}}Operation(), retval)
	return retval, err
}

`

const gorgoniaBinaryTemplateRaw = `// {{.FnName}} perfoms a pointwise {{lower .FnName}} operation.
{{if .AsSame -}}// retSame indicates if the data type of the return value should be the same as the input data type. It defaults to Bool otherwise.
{{end -}}
func {{.FnName}}(g *Graph, a, b node.Node{{if .AsSame}}, retSame bool{{end}}) (node.Node, error) { 
	retval := g.g.NewNode().(*engine.Node)
	g.g.AddNode(retval)
	g.g.SetWeightedEdge(g.g.NewWeightedEdge(retval, a, 1.0))
	g.g.SetWeightedEdge(g.g.NewWeightedEdge(retval, b, 2.0))
	{{if not .AsSame -}}
	err := g.g.ApplyOp(engine.New{{ .FnName}}Operation(nil,nil), retval)
	{{else -}}
	err := g.g.ApplyOp(engine.New{{ .FnName}}Operation(nil,nil,retSame), retval)
	{{end -}}
	return retval, err
}

`
