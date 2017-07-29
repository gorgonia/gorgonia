package main

import (
	"io"
	"reflect"
	"text/template"
)

type Signature struct {
	DocString      string
	Name           string
	NameTemplate   *template.Template
	ParamNames     []string
	ParamTemplates []*template.Template

	Kind reflect.Kind
	Err  bool
}

func (s *Signature) Write(w io.Writer) {
	s.NameTemplate.Execute(w, s)
	w.Write([]byte("("))
	for i, p := range s.ParamTemplates {
		w.Write([]byte(s.ParamNames[i]))
		w.Write([]byte(" "))
		p.Execute(w, s.Kind)

		if i < len(s.ParamNames) {
			w.Write([]byte(", "))
		}
	}
	w.Write([]byte(")"))
	if s.Err {
		w.Write([]byte("(err error)"))
	}
}

const (
	golinkPragmaRaw = "//go:linkname {{.Name}}{{short .Kind}} github.com/chewxy/{{vecPkg .Kind}}{{getalias .Name}}\n"

	typeAnnotatedNameRaw = `{{.Name}}{{short .Kind}}`
	plainNameRaw         = `{{.Name}}`
)

const (
	scalarTypeRaw    = `{{asType .}}`
	sliceTypeRaw     = `[]{{asType .}}`
	iteratorTypeRaw  = `Iterator`
	interfaceTypeRaw = "interface{}"
	boolsTypeRaw     = `[]bool`
	boolTypeRaw      = `bool`
	intTypeRaw       = `int`
	reflectTypeRaw   = `reflect.Type`

	// arrayTypeRaw        = `Array`
	arrayTypeRaw            = `*storage.Header`
	unaryFuncTypeRaw        = `func({{asType .}}){{asType .}} `
	unaryFuncErrTypeRaw     = `func({{asType .}}) ({{asType .}}, error)`
	reductionFuncTypeRaw    = `func(a, b {{asType .}}) {{asType .}}`
	reductionFuncTypeErrRaw = `func(a, b {{asType .}}) ({{asType .}}, error)`
	tensorTypeRaw           = `Tensor`
	splatFuncOptTypeRaw     = `...FuncOpt`
	denseTypeRaw            = `*Dense`
)

var (
	golinkPragma      *template.Template
	typeAnnotatedName *template.Template
	plainName         *template.Template

	scalarType       *template.Template
	sliceType        *template.Template
	iteratorType     *template.Template
	interfaceType    *template.Template
	boolsType        *template.Template
	boolType         *template.Template
	intType          *template.Template
	reflectType      *template.Template
	arrayType        *template.Template
	unaryFuncType    *template.Template
	unaryFuncErrType *template.Template
	tensorType       *template.Template
	splatFuncOptType *template.Template
	denseType        *template.Template
)

func init() {
	golinkPragma = template.Must(template.New("golinkPragma").Funcs(funcs).Parse(golinkPragmaRaw))
	typeAnnotatedName = template.Must(template.New("type annotated name").Funcs(funcs).Parse(typeAnnotatedNameRaw))
	plainName = template.Must(template.New("plainName").Funcs(funcs).Parse(plainNameRaw))

	scalarType = template.Must(template.New("scalarType").Funcs(funcs).Parse(scalarTypeRaw))
	sliceType = template.Must(template.New("sliceType").Funcs(funcs).Parse(sliceTypeRaw))
	iteratorType = template.Must(template.New("iteratorType").Funcs(funcs).Parse(iteratorTypeRaw))
	interfaceType = template.Must(template.New("interfaceType").Funcs(funcs).Parse(interfaceTypeRaw))
	boolsType = template.Must(template.New("boolsType").Funcs(funcs).Parse(boolsTypeRaw))
	boolType = template.Must(template.New("boolType").Funcs(funcs).Parse(boolTypeRaw))
	intType = template.Must(template.New("intTYpe").Funcs(funcs).Parse(intTypeRaw))
	reflectType = template.Must(template.New("reflectType").Funcs(funcs).Parse(reflectTypeRaw))
	arrayType = template.Must(template.New("arrayType").Funcs(funcs).Parse(arrayTypeRaw))
	unaryFuncType = template.Must(template.New("unaryFuncType").Funcs(funcs).Parse(unaryFuncTypeRaw))
	unaryFuncErrType = template.Must(template.New("unaryFuncErrType").Funcs(funcs).Parse(unaryFuncErrTypeRaw))
	tensorType = template.Must(template.New("tensorType").Funcs(funcs).Parse(tensorTypeRaw))
	splatFuncOptType = template.Must(template.New("splatFuncOpt").Funcs(funcs).Parse(splatFuncOptTypeRaw))
	denseType = template.Must(template.New("*Dense").Funcs(funcs).Parse(denseTypeRaw))
}
