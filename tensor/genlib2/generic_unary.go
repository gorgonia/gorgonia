package main

import (
	"io"
	"text/template"
)

type GenericUncondUnary struct {
	TypedUnaryOp
	Iter bool
}

func (fn *GenericUncondUnary) Name() string {
	if fn.Iter {
		return fn.TypedUnaryOp.Name() + "Iter"
	}
	return fn.TypedUnaryOp.Name()
}

func (fn *GenericUncondUnary) Signature() *Signature {
	paramNames := []string{"a"}
	paramTemplates := []*template.Template{sliceType}
	var err bool
	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
		err = true
	}
	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind(),
		Err:  err,
	}
}

func (fn *GenericUncondUnary) WriteBody(w io.Writer) {
	var IterName0 string
	T := template.New(fn.Name()).Funcs(funcs)

	if fn.Iter {
		T = template.Must(T.Parse(genericUnaryIterLoopRaw))
		IterName0 = "ait"
	} else {
		T = template.Must(T.Parse(genericLoopRaw))
	}
	template.Must(T.New("loopbody").Parse(basicSet))
	template.Must(T.New("opDo").Parse(unaryOpDo))
	template.Must(T.New("callFunc").Parse(unaryOpCallFunc))
	template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))
	template.Must(T.New("check").Parse(""))

	lb := LoopBody{
		TypedOp:   fn.TypedUnaryOp,
		Range:     "a",
		Left:      "a",
		Index0:    "i",
		IterName0: IterName0,
	}
	T.Execute(w, lb)
}

func (fn *GenericUncondUnary) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	if sig.Err {
		w.Write([]byte("\nreturn\n"))
	}
	w.Write([]byte("}\n\n"))
}

func generateGenericUncondUnary(f io.Writer, ak Kinds) {
	var gen []*GenericUncondUnary
	for _, tu := range typedUncondUnaries {
		if tc := tu.TypeClass(); tc != nil && !tc(tu.Kind()) {
			continue
		}
		fn := &GenericUncondUnary{
			TypedUnaryOp: tu,
		}
		gen = append(gen, fn)
	}

	for _, g := range gen {
		g.Write(f)
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
	}
}
