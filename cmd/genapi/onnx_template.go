package main

const onnxHeader = `package onnx 

import (
	"errors"

	"gonum.org/v1/gonum/graph"
	"gorgonia.org/gorgonia/internal/engine"
)

%v

`
const onnxUnaryTemplateRaw = `// {{.FnName}} performs a pointwise {{lower .FnName}}.
type {{.FnName}} struct{}

// Constructor to fulfil the interface ...
func (a *{{.FnName}}) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
		return engine.New{{.FnName}}Operation()(g, n.(*engine.Node))
	}
}

`

const onnxBinaryTemplateRaw = `// {{.FnName}} performs a pointwise {{lower .FnName}}.
type {{.FnName}} struct{}

// Constructor to fulfil the interface ...
func (a *{{.FnName}}) Constructor() func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
	return func(g graph.WeightedDirected, n graph.Node) (interface{}, error) {
		return engine.New{{.FnName}}Operation()(g, n.(*engine.Node))
	}
}

`
