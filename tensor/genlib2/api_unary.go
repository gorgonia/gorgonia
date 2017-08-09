package main

import (
	"io"
	"text/template"
)

type APIUnary struct {
	UnaryOp
}

func (fn *APIUnary) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	switch {
	case fn.UnaryOp.Name() == "Clamp":
		paramNames = []string{"a", "min", "max", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, interfaceType, splatFuncOptType}
	default:
		paramNames = []string{"a", "opts"}
		paramTemplates = []*template.Template{tensorType, splatFuncOptType}
	}
	return &Signature{
		Name:            fn.Name(),
		NameTemplate:    plainName,
		ParamNames:      paramNames,
		ParamTemplates:  paramTemplates,
		RetVals:         []string{"retVal"},
		RetValTemplates: []*template.Template{tensorType},
		Err:             true,
	}
}

func (fn *APIUnary) WriteBody(w io.Writer) {
	body := `var e Engine = a.Engine()
	if e == nil {
		e = StdEng{}
	}
	if {{interfaceName .Name | lower}}, ok := e.({{interfaceName .Name}}); ok {
		{{if eq .Name "Clamp" -}}
		return clamper.Clamp(a, min, max, opts...)
		{{else -}}
		return {{interfaceName .Name|lower}}.{{.Name}}(a, opts...)
		{{end -}}
	}
	err = errors.Errorf("Engine does not perform {{.Name}}")
	return
	`

	T := template.Must(template.New("body").Funcs(funcs).Parse(body))
	T.Execute(w, fn)
}

func (fn *APIUnary) Write(w io.Writer) {
	w.Write([]byte("func "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateUncondUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary
	for _, u := range unconditionalUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}

func generateCondUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary
	for _, u := range conditionalUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}

func generateSpecialUnaryAPI(f io.Writer, kinds Kinds) {
	var unaries []*APIUnary

	for _, u := range specialUnaries {
		fn := &APIUnary{
			UnaryOp: u,
		}
		unaries = append(unaries, fn)
	}
	for _, u := range unaries {
		u.Write(f)
	}
}
