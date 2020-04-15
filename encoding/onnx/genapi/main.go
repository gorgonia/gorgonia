package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"log"
	"strings"
	"text/template"
)

var allOps = []operation{
	{
		GorgonnxOp:    "hadamardProd",
		ONNXOpType:    "Mul",
		GorgoniaOp:    "HadamardProd",
		Arity:         2,
		Broadcastable: true,
	},
	{
		GorgonnxOp:    "hadamardDiv",
		ONNXOpType:    "Div",
		GorgoniaOp:    "HadamardDiv",
		Arity:         2,
		Broadcastable: true,
	},
	{
		ONNXOpType:    "Sub",
		Arity:         2,
		Broadcastable: true,
	},
	{
		ONNXOpType:    "Add",
		Arity:         2,
		Broadcastable: true,
	},
	{
		ONNXOpType: "Abs",
		Arity:      1,
	},
	{
		ONNXOpType: "Sign",
		Arity:      1,
	},
	{
		ONNXOpType: "Ceil",
		Arity:      1,
	},
	{
		ONNXOpType: "Floor",
		Arity:      1,
	},
	{
		ONNXOpType: "Sin",
		Arity:      1,
	},
	{
		ONNXOpType: "Cos",
		Arity:      1,
	},
	{
		ONNXOpType: "Exp",
		Arity:      1,
	},
	{
		// avoid log as it may conflict with the package
		GorgonnxOp: "logarithm",
		ONNXOpType: "Log",
		Arity:      1,
	},
	{
		ONNXOpType: "Log2",
		Arity:      1,
	},
	{
		ONNXOpType: "Relu",
		GorgoniaOp: "Rectify",
		Arity:      1,
	},
	{
		ONNXOpType: "Neg",
		Arity:      1,
	},
	{
		ONNXOpType: "Square",
		Arity:      1,
	},
	{
		ONNXOpType: "Sqrt",
		Arity:      1,
	},
	{
		ONNXOpType: "Inverse",
		Arity:      1,
	},
	{
		ONNXOpType: "Cube",
		Arity:      1,
	},
	{
		ONNXOpType: "Tanh",
		Arity:      1,
	},
	{
		ONNXOpType: "Sigmoid",
		Arity:      1,
	},
	{
		ONNXOpType: "Log1p",
		Arity:      1,
	},
	{
		ONNXOpType: "Expm1",
		Arity:      1,
	},
	{
		ONNXOpType: "Softplus",
		Arity:      1,
	}}

func main() {
	test := flag.Bool("test", false, "generate test file")
	flag.Parse()
	var t *template.Template
	if *test {
		t = testTmpl
		fmt.Println(testHeader)
	} else {
		t = opTmpl
		fmt.Println(opHeader)
	}
	for _, op := range allOps {
		if op.GorgonnxOp == "" {
			op.GorgonnxOp = strings.ToLower(op.ONNXOpType)
		}
		if op.GorgoniaOp == "" {
			op.GorgoniaOp = op.ONNXOpType
		}

		var buf bytes.Buffer
		if err := t.Execute(&buf, op); err != nil {
			log.Fatal(err)
		}
		p, err := format.Source(buf.Bytes())
		if err != nil {
			log.Fatal("Cannot format", err)
		}
		fmt.Println(string(p))
	}
}
