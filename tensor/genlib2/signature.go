package main

import (
	"io"
	"reflect"
	"text/template"
)

type Signature struct {
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
	scalarTypeRaw   = `{{asType .}}`
	sliceTypeRaw    = `[]{{asType .}}`
	iteratorTypeRaw = `Iterator`
	boolsTypeRaw    = `[]bool`
	reflectTypeRaw  = `reflect.Type`
	arrayTypeRaw    = `Array`
)

var (
	golinkPragma      *template.Template
	typeAnnotatedName *template.Template
	plainName         *template.Template

	scalarType   *template.Template
	sliceType    *template.Template
	iteratorType *template.Template
	boolsType    *template.Template
	reflectType  *template.Template
	arrayType    *template.Template
)

func init() {
	golinkPragma = template.Must(template.New("golinkPragmat").Funcs(funcs).Parse(golinkPragmaRaw))
	typeAnnotatedName = template.Must(template.New("type annotated name").Funcs(funcs).Parse(typeAnnotatedNameRaw))
	plainName = template.Must(template.New("plainName").Funcs(funcs).Parse(plainNameRaw))

	scalarType = template.Must(template.New("scalarType").Funcs(funcs).Parse(scalarTypeRaw))
	sliceType = template.Must(template.New("sliceType").Funcs(funcs).Parse(sliceTypeRaw))
	iteratorType = template.Must(template.New("iteratorType").Funcs(funcs).Parse(iteratorTypeRaw))
	boolsType = template.Must(template.New("boolsType").Funcs(funcs).Parse(boolsTypeRaw))
	reflectType = template.Must(template.New("reflectType").Funcs(funcs).Parse(reflectTypeRaw))
	arrayType = template.Must(template.New("arrayType").Funcs(funcs).Parse(arrayTypeRaw))
}
