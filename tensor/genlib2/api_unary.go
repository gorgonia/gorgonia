package main

import (
	"io"
	"text/template"
)

type APIUnary struct {
	UnaryOp
}

func (fn *APIUnary) Signature() *Signature {
	return &Signature{
		Name:            fn.Name(),
		NameTemplate:    plainName,
		ParamNames:      []string{"a", "opts"},
		ParamTemplates:  []*template.Template{tensorType, splatFuncOptType},
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
	if {{.Name|lower}}er, ok := e.({{.Name}}er); ok {
		return {{.Name|lower}}er.{{.Name}}(a, opts...)
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
