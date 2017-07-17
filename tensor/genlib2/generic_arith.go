package main

import (
	"io"
	"text/template"
)

type GenericVecVec struct {
	TypedBinOp
	RequiresIterator bool
	Incr             bool
	Check            TypeClass // can be nil
	CheckTemplate    string
}

func (fn *GenericVecVec) Signature() *Signature {
	var nameTemplate *template.Template
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.RequiresIterator && fn.Incr:
		nameTemplate = vvNameIncrIter
		paramNames = []string{"a", "b", "incr", "ait", "bit", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType, iteratorType}
		err = true
	case fn.RequiresIterator && !fn.Incr:
		nameTemplate = vvNameIter
		paramNames = []string{"a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType, iteratorType}
		err = true
	case !fn.RequiresIterator && fn.Incr:
		nameTemplate = vvNameIncr
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
	default:
		nameTemplate = vvName
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   nameTemplate,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind,
		Err:  err,
	}
}

func (fn *GenericVecVec) WriteBody(w io.Writer) {
	var Range, Left, Right string
	var T *template.Template

	Range = "a"
	switch {
	case fn.RequiresIterator && fn.Incr:
		Range = "incr"
		Left = "a[i]"
		Right = "b[j]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(vvIterIncrLoopRaw))
	case fn.RequiresIterator && !fn.Incr:
		Left = "a[i]"
		Right = "b[j]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(vvIterLoopRaw))
	case !fn.RequiresIterator && fn.Incr:
		Range = "incr"
		Left = "a[i]"
		Right = "b[i]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(vvIncrLoopRaw))
	default:
		Left = "a[i]"
		Right = "b[i]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(vvLoopRaw))
	}
	template.Must(T.New("symbol").Funcs(funcs).Parse(fn.SymbolTemplate()))

	if fn.Check != nil && fn.Check(fn.Kind) {
		w.Write([]byte("var errs errorIndices\n"))
	}
	template.Must(T.New("check").Funcs(funcs).Parse(fn.CheckTemplate))

	lb := LoopBody{
		TypedBinOp: fn.TypedBinOp,
		Range:      Range,
		Left:       Left,
		Right:      Right,
	}
	T.Execute(w, lb)
}

func (fn *GenericVecVec) Write(w io.Writer) {
	sig := fn.Signature()
	if !fn.RequiresIterator && isFloat(fn.Kind) {
		golinkPragma.Execute(w, fn)
		w.Write([]byte("func "))
		sig.Write(w)
		w.Write([]byte("\n\n"))
		return
	}

	w.Write([]byte("func "))
	sig.Write(w)

	switch {
	case !fn.RequiresIterator && fn.Incr:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]; incr = incr[:len(a)]\n"))
	case !fn.RequiresIterator && !fn.Incr:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]\n"))
	default:
		w.Write([]byte("{"))
	}
	fn.WriteBody(w)
	if sig.Err {
		if fn.Check != nil {
			w.Write([]byte("\nif err != nil {\n return\n}\nerr = errs"))
		}
		w.Write([]byte("\nreturn\n"))
	}
	w.Write([]byte("}\n\n"))
}

type GenericMixed struct {
	GenericVecVec
	LeftVec bool
}

func (fn *GenericMixed) Signature() *Signature {
	var nameTemplate *template.Template
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr", "ait", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType}
		if fn.LeftVec {
			nameTemplate = vsNameIncrIter
			paramTemplates[1] = scalarType
		} else {
			nameTemplate = svNameIncrIter
			paramTemplates[0] = scalarType
			paramNames[3] = "bit"
		}
		err = true
	case fn.RequiresIterator && !fn.Incr:
		paramNames = []string{"a", "b", "ait"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType}
		if fn.LeftVec {
			nameTemplate = vsNameIter
			paramTemplates[1] = scalarType
		} else {
			nameTemplate = svNameIter
			paramTemplates[0] = scalarType
			paramNames[2] = "bit"
		}

		err = true
	case !fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
		if fn.LeftVec {
			nameTemplate = vsNameIncr
			paramTemplates[1] = scalarType
		} else {
			nameTemplate = vsNameIncr
			paramTemplates[0] = scalarType
		}

	default:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
		if fn.LeftVec {
			nameTemplate = vsName
			paramTemplates[1] = scalarType
		} else {
			nameTemplate = vsName
			paramTemplates[0] = scalarType
		}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   nameTemplate,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind,
		Err:  err,
	}
}

func (fn *GenericMixed) Write(w io.Writer) {
	sig := fn.Signature()
	if !fn.RequiresIterator && isFloat(fn.Kind) {
		golinkPragma.Execute(w, fn)
		w.Write([]byte("func "))
		sig.Write(w)
		w.Write([]byte("\n\n"))
		return
	}

	w.Write([]byte("func "))
	sig.Write(w)

	switch {
	case !fn.RequiresIterator && fn.Incr:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]; incr = incr[:len(a)]\n"))
	case !fn.RequiresIterator && !fn.Incr:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]\n"))
	default:
		w.Write([]byte("{"))
	}
	fn.WriteBody(w)
	if sig.Err {
		if fn.Check != nil {
			w.Write([]byte("\nif err != nil {\n return\n}\nerr = errs"))
		}
		w.Write([]byte("\nreturn\n"))
	}
	w.Write([]byte("}\n\n"))
}

func makeGenericVecVecs(tbo []TypedBinOp) (retVal []*GenericVecVec) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind) {
			continue
		}
		fn := &GenericVecVec{
			TypedBinOp: tb,
		}
		if tb.Name() == "Div" && !isFloatCmplx(tb.Kind) {
			fn.Check = panicsDiv0
			fn.CheckTemplate = check0
		}
		retVal = append(retVal, fn)

	}

	return retVal
}

func makeGenericMixeds(tbo []TypedBinOp) (retVal []*GenericMixed) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && tc(tb.Kind) {
			continue
		}
		fn := &GenericMixed{
			GenericVecVec: GenericVecVec{
				TypedBinOp: tb,
			},
		}
		if tb.Name() == "Div" && !isFloatCmplx(tb.Kind) {
			fn.Check = panicsDiv0
			fn.CheckTemplate = check0
		}
		retVal = append(retVal, fn)
	}
	return
}

func generateGenericVecVec(w io.Writer, ak Kinds) {
	gen := makeGenericVecVecs(typedBinOps)

	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = false
		g.RequiresIterator = true
	}
	for _, g := range gen {
		g.Write(f)
		g.Incr = true
	}
	for _, g := range gen {
		g.Write(f)
	}
}

func generateGenericMixed(w io.Writer, ak Kinds) {
	gen := makeGenericMixeds(typedBinOps)

}
