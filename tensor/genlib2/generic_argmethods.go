package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

const argMethodLoopBody = `v := a[i]
if !set {
	f = v
	{{.ArgX}} = i 
	set = true
	continue
}
{{if isFloat .Kind -}}
if {{mathPkg .Kind}}IsNaN(v) || {{mathPkg .Kind}}IsInf(v, {{if eq .ArgX "min"}}-{{end}}1) {
	{{.ArgX}} = i
	return {{.ArgX}}
}
{{end -}}
if v {{if eq .ArgX "max"}}>{{else}}<{{end}} f {
	{{.ArgX}} = i
	f = v
}
`

const argMethodIter = `data := t.{{sliceOf .}}
tmp := make([]{{asType .}}, 0, lastSize)
for next, err = it.Next(); err == nil; next; err = it.Next() {
	tmp = append(tmp, data[next])
	if len(tmp) == lastSize {
		am := {{.ArgX | title}}(tmp)
		indices = append(indices, am)
		tmp = tmp[:0]
	}
}
return
`

type GenericArgMethod struct {
	ArgX   string
	Masked bool
	Range  string

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
		Kind:           fn.Kind,
	}
}

func (fn *GenericArgMethod) WriteBody(w io.Writer) {
	T := template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericLoopRaw))
	template.Must(T.New("loopbody").Parse(argMethodLoopBody))
	if fn.Masked {
		template.Must(T.New("check").Parse(maskCheck))
	} else {
		template.Must(T.New("check").Parse(""))
	}
	genericArgmaxVarDecl.Execute(w, fn)
	T.Execute(w, fn)
	fmt.Fprintf(w, "\nreturn %s\n", fn.ArgX)
}

func (fn *GenericArgMethod) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("int {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateGenericArgMethods(f io.Writer, ak Kinds) {
	var argMethods []*GenericArgMethod
	for _, k := range ak.Kinds {
		if !isOrd(k) {
			continue
		}
		m := &GenericArgMethod{
			ArgX:  "max",
			Kind:  k,
			Range: "a",
		}
		argMethods = append(argMethods, m)
	}

	// argmax
	for _, m := range argMethods {
		m.Write(f)
		m.Masked = true
	}

	for _, m := range argMethods {
		m.Write(f)
		m.Masked = false
		m.ArgX = "min"
	}
	// argmin
	for _, m := range argMethods {
		m.Write(f)
		m.Masked = true
	}

	for _, m := range argMethods {
		m.Write(f)
	}
}
