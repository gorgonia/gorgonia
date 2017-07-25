package main

import (
	"reflect"
	"text/template"
)

const argMethodLoopBody = `{{template "check" . -}}
if !set {
	f = v
	{{.ArgX}} = i 
	set = true
	continue
}
{{if isFloat .Kind -}}
if {{mathPkg .}}IsNaN(v) || {{mathPkg}}IsInf(v, 1) {
	{{.ArgX}} = i
	return {{.ArgX}}
}
{{end -}}
if v {{if eq .ArgX "max"}}>{{else}}<{{end}} f {
	{{.ArgX}} = i
	f = v
}
return {{.ArgX}}
`

// t :: arrayType; indices is retVal, with err
const iterable = `data := t.{{sliceOf .}}
tmp:=make([]{{asType .}}, 0, lastSize)
for next, err = it.Next(); err == nil; next; err = it.Next() {
	tmp = append(tmp, data[next])
	if len(tmp) == lastSize {
		am := {{.ArgX|title}}(tmp)
		indices = append(indices, am)
		tmp = tmp[:0]
	}
}
return
`

type GenericArgMethod struct {
	ArgX   string
	Masked bool
	Iter   bool

	Kind reflect.Kind
}

func (fn *GenericArgMethod) Name() string {
	switch {
	case fn.ArgX == "max" && fn.Masked:
		return "ArgmaxMasked"
	case fn.ArgX == "min" && fn.Masked:
		return "ArgminMasked"
	case fn.ArgX == "max" && !fn.Masked:
		return "Argmax"
	case fn.ArgX == "min" && !fn.Masked:
		return "Argmin"
	}
	panic("Unreachable")
}

func (fn *GenericArgMethod) Signature() *Signature {
	paramNames := []string{"a"}
	paramTemplates := []*template.Template{sliceType}

	if fn.Masked {
		paramNames = append(paramNames, "mask")
		paramTemplates = append(paramTemplates, boolsType)
	}
	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
	}
}
