package main

import (
	"io"
	"reflect"
	"text/template"
)

type EngineArith struct {
	Name           string
	VecVar         string
	PrepData       string
	TypeClassCheck string

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
		fn.VecVar = "a"
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "t"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "t"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	template.Must(prep.New("prep").Parse(arithPrepRaw))
	prep.Execute(w, fn)
	agg2Body.Execute(w, fn)
}

func (fn *EngineArith) Write(w io.Writer) {
	if tmpl, ok := arithDocStrings[fn.methName()]; ok {
		type tmp struct {
			Left, Right string
		}
		var ds tmp
		if fn.VV {
			ds.Left = "a"
			ds.Right = "b"
		} else {
			ds.Left = "t"
			ds.Right = "s"
		}
		tmpl.Execute(w, ds)
	}

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
			Name:           abo.Name(),
			VV:             true,
			TypeClassCheck: "Number",
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

type EngineCmp struct {
	Name           string
	VecVar         string
	PrepData       string
	TypeClassCheck string

	VV      bool
	LeftVec bool
}

func (fn *EngineCmp) methName() string {
	switch {
	case fn.VV:
		if fn.Name == "Eq" || fn.Name == "Ne" {
			return "El" + fn.Name
		}
		return fn.Name
	default:
		return fn.Name + "Scalar"
	}
}

func (fn *EngineCmp) Signature() *Signature {
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

func (fn *EngineCmp) WriteBody(w io.Writer) {
	var prep *template.Template
	switch {
	case fn.VV:
		prep = prepVV
		fn.VecVar = "a"
	case !fn.VV && fn.LeftVec:
		fn.VecVar = "t"
		fn.PrepData = "prepDataVS"
		prep = prepMixed
	default:
		fn.VecVar = "t"
		fn.PrepData = "prepDataSV"
		prep = prepMixed
	}
	template.Must(prep.New("prep").Parse(cmpPrepRaw))
	prep.Execute(w, fn)
	agg2CmpBody.Execute(w, fn)
}

func (fn *EngineCmp) Write(w io.Writer) {
	if tmpl, ok := cmpDocStrings[fn.methName()]; ok {
		type tmp struct {
			Left, Right string
		}
		var ds tmp
		if fn.VV {
			ds.Left = "a"
			ds.Right = "b"
		} else {
			ds.Left = "t"
			ds.Right = "s"
		}
		tmpl.Execute(w, ds)
	}
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("(retVal Tensor, err error) {\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateStdEngCmp(f io.Writer, ak Kinds) {
	var methods []*EngineCmp

	for _, abo := range cmpBinOps {
		var tc string
		if abo.Name() == "Eq" || abo.Name() == "Ne" {
			tc = "Eq"
		} else {
			tc = "Ord"
		}
		meth := &EngineCmp{
			Name:           abo.Name(),
			VV:             true,
			TypeClassCheck: tc,
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

type EngineUnary struct {
	Name           string
	TypeClassCheck string
	Kinds          []reflect.Kind
}

func (fn *EngineUnary) Signature() *Signature {
	return &Signature{
		Name:            fn.Name,
		NameTemplate:    plainName,
		ParamNames:      []string{"a", "opts"},
		ParamTemplates:  []*template.Template{tensorType, splatFuncOptType},
		RetVals:         []string{"retVal"},
		RetValTemplates: []*template.Template{tensorType},

		Err: true,
	}
}

func (fn *EngineUnary) WriteBody(w io.Writer) {
	prepUnary.Execute(w, fn)
	agg2UnaryBody.Execute(w, fn)
}

func (fn *EngineUnary) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func (e StdEng) "))
	sig.Write(w)
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("\n}\n"))
}

func generateStdEngUncondUnary(f io.Writer, ak Kinds) {
	tcc := []string{
		"Number",     // Neg
		"Number",     // Inv
		"Number",     // Square
		"Number",     // Cube
		"FloatCmplx", // Exp
		"FloatCmplx", // Tanhh
		"FloatCmplx", // Log
		"Float",      // Log2
		"FloatCmplx", // Log10
		"FloatCmplx", // Sqrt
		"Float",      // Cbrt
		"Float",      // InvSqrt
	}
	var gen []*EngineUnary
	for i, u := range unconditionalUnaries {
		var ks []reflect.Kind
		for _, k := range ak.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			ks = append(ks, k)
		}
		fn := &EngineUnary{
			Name:           u.Name(),
			TypeClassCheck: tcc[i],
			Kinds:          ks,
		}
		gen = append(gen, fn)
	}

	for _, fn := range gen {
		fn.Write(f)
	}
}

func generateStdEngCondUnary(f io.Writer, ak Kinds) {
	tcc := []string{
		"Signed", // Abs
		"Signed", // Sign
	}
	var gen []*EngineUnary
	for i, u := range conditionalUnaries {
		var ks []reflect.Kind
		for _, k := range ak.Kinds {
			if tc := u.TypeClass(); tc != nil && !tc(k) {
				continue
			}
			ks = append(ks, k)
		}
		fn := &EngineUnary{
			Name:           u.Name(),
			TypeClassCheck: tcc[i],
			Kinds:          ks,
		}
		gen = append(gen, fn)
	}

	for _, fn := range gen {
		fn.Write(f)
	}
}
