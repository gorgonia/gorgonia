package main

import (
	"fmt"
	"io"
	"os"
	"text/template"
)

type GenericVecVecCmp struct {
	TypedBinOp
	RetSame bool
	Iter    bool
}

func (fn *GenericVecVecCmp) Name() string {
	switch {
	case fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSameIter", fn.TypedBinOp.Name())
	case fn.Iter && !fn.RetSame:
		return fmt.Sprintf("%sIter", fn.TypedBinOp.Name())
	case !fn.Iter && fn.RetSame:
		return fmt.Sprintf("%sSame", fn.TypedBinOp.Name())
	default:
		return fn.TypedBinOp.Name()
	}
}

func (fn *GenericVecVecCmp) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.Iter && fn.RetSame:
		paramNames = []string{"a", "b", "ait", "bit"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType, iteratorType}
		err = true
	case fn.Iter && !fn.RetSame:
		paramNames = []string{"a", "b", "retVal", "ait", "bit", "rit"}
		paramTemplates = []*template.Template{sliceType, sliceType, boolsType, iteratorType, iteratorType, iteratorType}
		err = true
	case !fn.Iter && fn.RetSame:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
	default:
		paramNames = []string{"a", "b", "retVal"}
		paramTemplates = []*template.Template{sliceType, sliceType, boolsType}
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

func (fn *GenericVecVecCmp) WriteBody(w io.Writer) {
	var Range, Left, Right string
	var Index0, Index1, Index2 string
	var IterName0, IterName1, IterName2 string
	var T *template.Template

	Range = "a"
	Left = "a[i]"
	Right = "b[j]"
	Index0 = "i"
	Index1 = "j"
	switch {
	case fn.Iter && fn.RetSame:
		IterName0 = "ait"
		IterName1 = "bit"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericBinaryIterLoopRaw))
		template.Must(T.New("loopbody").Funcs(funcs).Parse(sameSet))
	case fn.Iter && !fn.RetSame:
		Range = "retVal"
		Index2 = "k"
		IterName0 = "ait"
		IterName1 = "bit"
		IterName2 = "rit"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericTernaryIterLoopRaw))
		template.Must(T.New("loopbody").Funcs(funcs).Parse(ternaryIterSet))
	case !fn.Iter && fn.RetSame:
		Right = "b[i]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Funcs(funcs).Parse(sameSet))
	default:
		Range = "retVal"
		Right = "b[i]"
		T = template.Must(template.New(fn.Name()).Funcs(funcs).Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Funcs(funcs).Parse(basicSet))
	}
	template.Must(T.New("opDo").Funcs(funcs).Parse(binOpDo))
	template.Must(T.New("callFunc").Funcs(funcs).Parse(""))
	template.Must(T.New("check").Funcs(funcs).Parse(""))
	template.Must(T.New("symbol").Funcs(funcs).Parse(fn.SymbolTemplate()))

	lb := LoopBody{
		TypedOp: fn.TypedBinOp,
		Range:   Range,
		Left:    Left,
		Right:   Right,

		Index0: Index0,
		Index1: Index1,
		Index2: Index2,

		IterName0: IterName0,
		IterName1: IterName1,
		IterName2: IterName2,
	}
	T.Execute(w, lb)
}

func (fn *GenericVecVecCmp) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	switch {
	case !fn.Iter && !fn.RetSame:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]; retVal=retVal[:len(a)]\n"))
	case !fn.Iter && fn.RetSame:
		w.Write([]byte("{\na = a[:len(a)]; b = b[:len(a)]\n"))
	default:
		w.Write([]byte("{"))
	}
	fn.WriteBody(w)
	if sig.Err {
		w.Write([]byte("\n return\n"))
	}
	w.Write([]byte("}\n\n"))
}

func makeGenericVecVecCmps(tbo []TypedBinOp) (retVal []*GenericVecVecCmp) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind()) {
			continue
		}
		fn := &GenericVecVecCmp{
			TypedBinOp: tb,
		}
		retVal = append(retVal, fn)
	}
	return
}

func generateGenericVecVecCmp(f io.Writer, ak Kinds) {
	gen := makeGenericVecVecCmps(typedCmps)
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true

	}
	fmt.Fprintln(os.Stderr, "Writing RETSAME = true")
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
		g.RetSame = false
		g.Iter = true
	}
	fmt.Fprintln(os.Stderr, "Writing RETSAME = false, Iter = true")
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	fmt.Fprintln(os.Stderr, "Writing RETSAME = true")
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
	}
}
