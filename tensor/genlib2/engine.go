package main

import (
	"io"
	"text/template"
)

type EngineArith struct {
	Name     string
	VecVar   string
	PrepData string

	VV      bool
	LeftVec bool
}

func (fn *EngineArith) methName() string {
	switch {
	case fn.VV:
		return fn.Name
	default:
		return fn.Name + "Scalar"
	}
}

func (fn *EngineArith) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.VV:
		paramNames = []string{"a", "b", "opts"}
		paramTemplates = []*template.Template{tensorType, tensorType, splatFuncOptType}
	default:
		paramNames = []string{"t", "s", "leftTensor", "opts"}
		paramTemplates = []*template.Template{tensorType, interfaceType, boolType, splatFuncOptType}
	}
	return &Signature{
		Name:           fn.methName(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            false,
	}
}

func (fn *EngineArith) WriteBody(w io.Writer) {
	var prep *template.Template
	switch {
	case fn.VV:
		prep = prepVV
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "b"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "a"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	prep.Execute(w, fn)
	agg2Body.Execute(w, fn)
}

func (fn *EngineArith) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("(retVal Tensor, err error) {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateStdEngArith(f io.Writer, ak Kinds) {
	var methods []*EngineArith
	for _, abo := range arithBinOps {
		meth := &EngineArith{
			Name: abo.Name(),
			VV:   true,
		}
		methods = append(methods, meth)
	}

	// VV
	for _, meth := range methods {
		meth.Write(f)
		meth.VV = false
	}

	// Scalar
	for _, meth := range methods {
		meth.Write(f)
		meth.LeftVec = true
	}

}
