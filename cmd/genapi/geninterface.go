package main

import (
	"go/parser"
	"go/token"
	"io"
	"log"
	"path"
	"strings"
	"text/template"
)

type UnaryOpInterfaceData struct {
	OpTypes []string
	Dtype   string // f32, f64
}

const unaryOpInterfaceRaw = `func (f *s{{.Dtype}}UnaryOperator) unaryOpType() ʘUnaryOperatorType {
	{{$dt := .Dtype -}}
	switch f {
		{{range $i, $op := .OpTypes -}}
		case &{{$op}}{{$dt}}:
			return {{$op}}OpType
		{{end -}}
	}
	return maxʘUnaryOperator
}

func (f *s{{.Dtype}}UnaryOperator) String() string { return f.unaryOpType().String() }

`

var unaryOpInterface *template.Template

func init() {
	unaryOpInterface = template.Must(template.New("UnOpInterface").Funcs(funcmap).Parse(unaryOpInterfaceRaw))
}

func generateUnaryInterface(outFile io.Writer) {
	// parse operator_unary_const.go
	filename := path.Join(gorgonialoc, unaryOps)
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, filename, nil, parser.AllErrors)
	if err != nil {
		log.Fatal(err)
	}

	unaryNames := constTypes(file.Decls, "ʘUnaryOperatorType", "maxʘUnaryOperator")
	var opNames []string
	for _, v := range unaryNames {
		op := strings.TrimSuffix(v, "OpType")
		opNames = append(opNames, op)
	}

	dtypes := []string{"f32", "f64"}
	for _, dt := range dtypes {
		data := UnaryOpInterfaceData{opNames, dt}
		unaryOpInterface.Execute(outFile, data)
	}
}
