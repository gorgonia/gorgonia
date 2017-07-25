package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type InternalEngArithMethod struct {
	BinOp
	Kinds []reflect.Kind
	Incr  bool
	Iter  bool
}

type eLoopBody struct {
	BinOp

	Err   bool
	Kinds []reflect.Kind
}

func (fn *InternalEngArithMethod) Name() string {
	switch {
	case fn.Incr && fn.Iter:
		return fmt.Sprintf("%sIterIncr", fn.BinOp.Name())
	case fn.Incr && !fn.Iter:
		return fmt.Sprintf("%sIncr", fn.BinOp.Name())
	case !fn.Incr && fn.Iter:
		return fmt.Sprintf("%sIter", fn.BinOp.Name())
	default:
		return fn.BinOp.Name()
	}
}

func (fn *InternalEngArithMethod) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	switch {
	case fn.Iter && fn.Incr:
		paramNames = []string{"t", "a", "b", "incr", "ait", "bit", "iit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType, iteratorType, iteratorType, iteratorType}
	case fn.Iter && !fn.Incr:
		paramNames = []string{"t", "a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, iteratorType, iteratorType}
	case !fn.Iter && fn.Incr:
		paramNames = []string{"t", "a", "b", "incr"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, arrayType}
	default:
		paramNames = []string{"t", "a", "b"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType}

	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            true,
	}
}

func (fn *InternalEngArithMethod) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Incr && !fn.Iter:
		T = eArithIncr
	case fn.Iter && !fn.Incr:
		T = eArithIter
	default:
		T = eArith
	}
	lb := eLoopBody{
		BinOp: fn.BinOp,
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngArithMethod) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateEArith(f io.Writer, kinds Kinds) {
	var methods []*InternalEngArithMethod
	for _, bo := range arithBinOps {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := bo.TypeClass(); tc != nil && tc(k) {
				ks = append(ks, k)
			}
		}
		meth := &InternalEngArithMethod{
			BinOp: bo,
			Kinds: ks,
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = true
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = false
		meth.Iter = true
	}
	for _, meth := range methods {
		meth.Write(f)
		meth.Incr = true
	}

	for _, meth := range methods {
		meth.Write(f)
	}
}

/* MAP */

type InternalEngMap struct {
	Kinds []reflect.Kind
	Iter  bool
}

func (fn *InternalEngMap) Signature() *Signature {
	paramNames := []string{"t", "fn", "a", "incr"}
	paramTemplates := []*template.Template{reflectType, interfaceType, arrayType, boolType}
	name := "Map"
	if fn.Iter {
		paramNames = append(paramNames, "ait")
		paramTemplates = append(paramTemplates, iteratorType)
		name += "Iter"
	}

	return &Signature{
		Name:           name,
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,
		Err:            true,
	}
}

func (fn *InternalEngMap) WriteBody(w io.Writer) {
	T := eMap
	if fn.Iter {
		T = eMapIter
	}
	template.Must(T.New("fntype0").Funcs(funcs).Parse(unaryFuncTypeRaw))
	template.Must(T.New("fntype1").Funcs(funcs).Parse(unaryFuncErrTypeRaw))

	lb := eLoopBody{
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngMap) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)

	w.Write([]byte("\nreturn\n}\n\n"))
}

func generateEMap(f io.Writer, kinds Kinds) {
	m := new(InternalEngMap)
	for _, k := range kinds.Kinds {
		if !isAddable(k) {
			continue
		}
		m.Kinds = append(m.Kinds, k)
	}
	m.Write(f)
	m.Iter = true
	m.Write(f)
}

/* Cmp */

// InternalEngCmpMethod is exactly the same structure as the arith one, except it's Same instead of Incr.
// Some copy and paste leads to more clarity, rather than reusing the structure.
type InternalEngCmp struct {
	BinOp
	Kinds   []reflect.Kind
	RetSame bool
	Iter    bool
}

func (fn *InternalEngCmp) Name() string {
	switch {
	case fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSameIter", fn.BinOp.Name())
	case fn.Iter && !fn.RetSame:
		return fmt.Sprintf("%sIter", fn.BinOp.Name())
	case !fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSame", fn.BinOp.Name())
	default:
		return fn.BinOp.Name()
	}
}

func (fn *InternalEngCmp) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template

	switch {
	case fn.Iter && fn.RetSame:
		paramNames = []string{"t", "a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, iteratorType, iteratorType}
	case fn.Iter && !fn.RetSame:
		paramNames = []string{"t", "a", "b", "retVal", "ait", "bit", "rit"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, boolsType, iteratorType, iteratorType, iteratorType}
	case !fn.Iter && fn.RetSame:
		paramNames = []string{"t", "a", "b"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType}
	default:
		paramNames = []string{"t", "a", "b", "retVal"}
		paramTemplates = []*template.Template{reflectType, arrayType, arrayType, boolsType}
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   plainName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Err: true,
	}
}

func (fn *InternalEngCmp) WriteBody(w io.Writer) {
	var T *template.Template
	switch {
	case fn.Iter && fn.RetSame:
		T = eCmpSameIter
	case fn.Iter && !fn.RetSame:
		T = eCmpBoolIter
	case !fn.Iter && fn.RetSame:
		T = eCmpSame
	default:
		T = eCmpBool
	}

	lb := eLoopBody{
		BinOp: fn.BinOp,
		Kinds: fn.Kinds,
	}
	T.Execute(w, lb)
}

func (fn *InternalEngCmp) Write(w io.Writer) {
	w.Write([]byte("func (e E) "))
	sig := fn.Signature()
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n\n"))
}

func generateECmp(f io.Writer, kinds Kinds) {
	var methods []*InternalEngCmp
	for _, bo := range cmpBinOps {
		var ks []reflect.Kind
		for _, k := range kinds.Kinds {
			if tc := bo.TypeClass(); tc != nil && tc(k) {
				ks = append(ks, k)
			}
		}
		meth := &InternalEngCmp{
			BinOp: bo,
			Kinds: ks,
		}
		methods = append(methods, meth)
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = true
	}

	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = false
		meth.Iter = true
	}
	for _, meth := range methods {
		meth.Write(f)
		meth.RetSame = true
	}

	for _, meth := range methods {
		meth.Write(f)
	}
}
