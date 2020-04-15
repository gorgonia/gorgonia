package main

import "text/template"

var opTmpl = template.Must(template.New("op").Funcs(iterator).Parse(opTemplate))
var testTmpl = template.Must(template.New("test").Parse(testTemplate))

type operation struct {
	GorgonnxOp    string
	ONNXOpType    string
	GorgoniaOp    string
	Arity         int
	Broadcastable bool
}

const opTemplate = `
type {{ .GorgonnxOp }} struct{}

func init() {
	register("{{ .ONNXOpType }}", new{{ .GorgonnxOp }})
}

func new{{ .GorgonnxOp }}() operator {
	return &{{ .GorgonnxOp }}{}
}

func (a *{{ .GorgonnxOp }}) apply(g *Graph, n *Node) error {
	children := getOrderedChildren(g.g, n)
	err := checkCondition(children, {{ .Arity }})
	if err != nil {
		return err
	}
	{{ if .Broadcastable }}
	x, y, err := broadcast(children[0], children[1])
	if err != nil {
		err, ok := err.(*onnx.ErrNotImplemented)
		if ok {
			err.Operator = "{{ .ONNXOpType }} / {{ .GorgonnxOp }}"
		}
		return err
	}
	n.gorgoniaNode, err = gorgonia.{{ .GorgoniaOp }}(x,y)
	{{ else }}
	n.gorgoniaNode, err = gorgonia.{{ .GorgoniaOp }}(
		{{- range $val := Iterate .Arity }}
		children[{{ $val }}].gorgoniaNode, 
		{{- end }}
	)
	{{ end }}
	return err
}

func (a *{{ .GorgonnxOp }}) init(o onnx.Operation) error {
	return nil
}
`

const testTemplate = `
// Test{{ .ONNXOpType }} ...
func Test{{ .ONNXOpType }}(t *testing.T) {
	  for _, tc := range testbackend.GetOpTypeTests("{{ .ONNXOpType }}") {
		  tc := tc() // capture range variable
		  t.Run(tc.GetInfo(), tc.RunTest(NewGraph(), true))
	  }
}
`

var iterator = template.FuncMap{
	"Iterate": func(count int) []int {
		var i int
		var Items []int
		for i = 0; i < count; i++ {
			Items = append(Items, i)
		}
		return Items
	},
}

const testHeader = `package gorgonnx

import (
	"testing"

	"github.com/owulveryck/onnx-go/backend/testbackend"
	_ "github.com/owulveryck/onnx-go/backend/testbackend/onnx"
)`

const opHeader = `package gorgonnx

import (
	"github.com/owulveryck/onnx-go"
	"gorgonia.org/gorgonia"
)`
