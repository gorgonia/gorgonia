package main

import (
	"fmt"
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

func (fn *GenericVecVec) Name() string {
	switch {
	case fn.RequiresIterator && fn.Incr:
		return fmt.Sprintf("%sIncrIter", fn.TypedBinOp.Name())
	case fn.RequiresIterator && !fn.Incr:
		return fmt.Sprintf("%sIter", fn.TypedBinOp.Name())
	case !fn.RequiresIterator && fn.Incr:
		return fmt.Sprintf("%sIncr", fn.TypedBinOp.Name())
	default:
		return fn.TypedBinOp.Name()
	}
}

func (fn *GenericVecVec) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr", "ait", "bit", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType, iteratorType}
		err = true
	case fn.RequiresIterator && !fn.Incr:
		paramNames = []string{"a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType, iteratorType}
		err = true
	case !fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
	default:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
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
		T = template.Must(template.New("vvIterIncrLoop").Funcs(funcs).Parse(vvIterIncrLoopRaw))
	case fn.RequiresIterator && !fn.Incr:
		Left = "a[i]"
		Right = "b[j]"
		T = template.Must(template.New("vvIterLoop").Funcs(funcs).Parse(vvIterLoopRaw))
	case !fn.RequiresIterator && fn.Incr:
		Range = "incr"
		Left = "a[i]"
		Right = "b[i]"
		T = template.Must(template.New("vvIncrLoop").Funcs(funcs).Parse(vvIncrLoopRaw))
	default:
		Left = "a[i]"
		Right = "b[i]"
		T = template.Must(template.New("vvLoop").Funcs(funcs).Parse(vvLoopRaw))
	}
	template.Must(T.New("callFunc").Funcs(funcs).Parse(callFunc))
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

func (fn *GenericMixed) Name() string {
	n := fn.GenericVecVec.Name()
	if fn.LeftVec {
		n += "VS"
	} else {
		n += "SV"
	}
	return n
}

func (fn *GenericMixed) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr", "ait", "iit"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType, iteratorType, iteratorType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
			paramNames[3] = "bit"
		}
		err = true
	case fn.RequiresIterator && !fn.Incr:
		paramNames = []string{"a", "b", "ait"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
			paramNames[2] = "bit"
		}

		err = true
	case !fn.RequiresIterator && fn.Incr:
		paramNames = []string{"a", "b", "incr"}
		paramTemplates = []*template.Template{sliceType, sliceType, sliceType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
		}

	default:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
		if fn.LeftVec {
			paramTemplates[1] = scalarType
		} else {
			paramTemplates[0] = scalarType
		}
	}

	if fn.Check != nil {
		err = true
	}

	return &Signature{
		Name:           fn.Name(),
		NameTemplate:   typeAnnotatedName,
		ParamNames:     paramNames,
		ParamTemplates: paramTemplates,

		Kind: fn.Kind,
		Err:  err,
	}
}

func (fn *GenericMixed) WriteBody(w io.Writer) {
	var Range, Left, Right, IterName string
	var T *template.Template

	Range = "a"
	Left = "a[i]"
	Right = "b[i]"
	switch {
	case fn.RequiresIterator && fn.Incr:
		Range = "incr"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(mixedIterIncrLoopRaw))
	case fn.RequiresIterator && !fn.Incr:
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(mixedIterLoopRaw))
	case !fn.RequiresIterator && fn.Incr:
		Range = "incr"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(mixedIncrLoopRaw))
	default:
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(mixedLoopRaw))
	}

	if fn.LeftVec {
		Right = "b"
	} else {
		Left = "a"
		if !fn.Incr {
			Range = "b"
		}
	}

	switch {
	case fn.RequiresIterator && fn.Incr && fn.LeftVec:
		IterName = "ait"
	case fn.RequiresIterator && !fn.Incr && fn.LeftVec:
		IterName = "ait"
	case fn.RequiresIterator && fn.Incr && !fn.LeftVec:
		IterName = "bit"
	case fn.RequiresIterator && !fn.Incr && !fn.LeftVec:
		IterName = "bit"
	}

	template.Must(T.New("callFunc").Funcs(funcs).Parse(callFunc))
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
		IterName:   IterName,
	}
	T.Execute(w, lb)
}

func (fn *GenericMixed) Write(w io.Writer) {
	sig := fn.Signature()

	w.Write([]byte("func "))
	sig.Write(w)

	w.Write([]byte("{\n"))

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
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind) {
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

func generateGenericVecVecArith(f io.Writer, ak Kinds) {
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

func generateGenericMixedArith(f io.Writer, ak Kinds) {
	gen := makeGenericMixeds(typedBinOps)

	// SV first
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

		// reset
		g.LeftVec = true
		g.Incr = false
		g.RequiresIterator = false
	}

	// VS
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
