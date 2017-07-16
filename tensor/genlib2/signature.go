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
	golinkPragmaRaw = "//go:linkname {{.Name}}{{short .Kind}} {{vecPkg .Kind}}{{if .Incr}}Incr{{end}}{{.Name}}\n"

	vvNameRaw         = `{{.Name}}{{short .Kind}}`
	vvNameIncrRaw     = `{{.Name}}Incr{{short .Kind}}`
	vvNameIterRaw     = `{{.Name}}Iter{{short .Kind}}`
	vvNameIncrIterRaw = `{{.Name}}IncrIter{{short .Kind}}`

	svNameRaw     = `{{.Name}}SV{{short .Kind}}`
	svNameIterRaw = `{{.Name}}IterSV{{short .Kind}}`

	vsNameRaw     = `{{.Name}}VS{{short.Kind}}`
	vsNameIterRaw = `{{.Name}}IterVS{{short.Kind}}`
)

const (
	scalarTypeRaw   = `{{asType .}}`
	sliceTypeRaw    = `[]{{asType .}}`
	iteratorTypeRaw = `Iterator`
	boolsTypeRaw    = `[]bool`
)

var (
	golinkPragma *template.Template

	vvName         *template.Template
	vvNameIter     *template.Template
	vvNameIncr     *template.Template
	vvNameIncrIter *template.Template

	svName     *template.Template
	svNameIter *template.Template

	vsName     *template.Template
	vsNameIter *template.Template

	scalarType   *template.Template
	sliceType    *template.Template
	iteratorType *template.Template
	boolsType    *template.Template
)

func init() {
	golinkPragma = template.Must(template.New("golinkPragmat").Funcs(funcs).Parse(golinkPragmaRaw))
	vvName = template.Must(template.New("vvName").Funcs(funcs).Parse(vvNameRaw))
	vvNameIter = template.Must(template.New("vvName iter").Funcs(funcs).Parse(vvNameIterRaw))
	vvNameIncr = template.Must(template.New("vvName incr").Funcs(funcs).Parse(vvNameIncrRaw))
	vvNameIncrIter = template.Must(template.New("vvName incr iter").Funcs(funcs).Parse(vvNameIncrIterRaw))

	svName = template.Must(template.New("svName").Funcs(funcs).Parse(svNameRaw))
	svNameIter = template.Must(template.New("svName iter").Funcs(funcs).Parse(svNameIterRaw))
	vsName = template.Must(template.New("vsName").Funcs(funcs).Parse(vsNameRaw))
	vsNameIter = template.Must(template.New("vsName iter").Funcs(funcs).Parse(vsNameIterRaw))

	scalarType = template.Must(template.New("scalarType").Funcs(funcs).Parse(scalarTypeRaw))
	sliceType = template.Must(template.New("sliceType").Funcs(funcs).Parse(sliceTypeRaw))
	iteratorType = template.Must(template.New("iteratorType").Funcs(funcs).Parse(iteratorTypeRaw))
	boolsType = template.Must(template.New("boolsType").Funcs(funcs).Parse(boolsTypeRaw))
}
