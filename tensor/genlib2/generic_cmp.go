package main

import (
	"fmt"
	"io"
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

type GenericMixedCmp struct {
	GenericVecVecCmp
	LeftVec bool
}

func (fn *GenericMixedCmp) Name() string {
	n := fn.GenericVecVecCmp.Name()
	if fn.LeftVec {
		n += "VS"
	} else {
		n += "SV"
	}
	return n
}

func (fn *GenericMixedCmp) Signature() *Signature {
	var paramNames []string
	var paramTemplates []*template.Template
	var err bool

	switch {
	case fn.Iter && !fn.RetSame:
		paramNames = []string{"a", "b", "retVal", "ait", "rit"}
		paramTemplates = []*template.Template{sliceType, sliceType, boolsType, iteratorType, iteratorType}
		err = true
	case fn.Iter && fn.RetSame:
		paramNames = []string{"a", "b", "ait"}
		paramTemplates = []*template.Template{sliceType, sliceType, iteratorType}
		err = true
	case !fn.Iter && fn.RetSame:
		paramNames = []string{"a", "b"}
		paramTemplates = []*template.Template{sliceType, sliceType}
	default:
		paramNames = []string{"a", "b", "retVal"}
		paramTemplates = []*template.Template{sliceType, sliceType, boolsType}
	}
	if fn.LeftVec {
		paramTemplates[1] = scalarType
	} else {
		paramTemplates[0] = scalarType
		if fn.Iter && !fn.RetSame {
			paramNames[3] = "bit"
		} else if fn.Iter && fn.RetSame {
			paramNames[2] = "bit"
		}
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

func (fn *GenericMixedCmp) WriteBody(w io.Writer) {
	var Range, Left, Right string
	var Index0, Index1 string
	var IterName0, IterName1 string
	var T *template.Template

	Range = "a"
	Left = "a[i]"
	Right = "b[i]"
	Index0 = "i"

	T = template.New(fn.Name()).Funcs(funcs)
	switch {
	case fn.Iter && !fn.RetSame:
		Range = "retVal"
		T = template.Must(T.Parse(genericBinaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(ternaryIterSet))
	case fn.Iter && fn.RetSame:
		T = template.Must(T.Parse(genericUnaryIterLoopRaw))
		template.Must(T.New("loopbody").Parse(sameSet))
	case !fn.Iter && fn.RetSame:
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(sameSet))
	default:
		T = template.Must(T.Parse(genericLoopRaw))
		template.Must(T.New("loopbody").Parse(basicSet))
	}

	if fn.LeftVec {
		Right = "b"
	} else {
		Left = "a"
	}
	if !fn.RetSame {
		Range = "retVal"
	} else {
		if !fn.LeftVec {
			Range = "b"
		}
	}

	switch {
	case fn.Iter && !fn.RetSame && fn.LeftVec:
		IterName0 = "ait"
		IterName1 = "rit"
		Index1 = "k"
	case fn.Iter && fn.RetSame && fn.LeftVec:
		IterName0 = "ait"
	case fn.Iter && !fn.RetSame && !fn.LeftVec:
		IterName0 = "bit"
		IterName1 = "rit"
		Index1 = "k"
	case fn.Iter && fn.RetSame && !fn.LeftVec:
		IterName0 = "bit"
	}

	template.Must(T.New("callFunc").Parse(""))
	template.Must(T.New("opDo").Parse(binOpDo))
	template.Must(T.New("symbol").Parse(fn.SymbolTemplate()))
	template.Must(T.New("check").Parse(""))

	lb := LoopBody{
		TypedOp: fn.TypedBinOp,
		Range:   Range,
		Left:    Left,
		Right:   Right,

		Index0:    Index0,
		Index1:    Index1,
		IterName0: IterName0,
		IterName1: IterName1,
	}
	T.Execute(w, lb)
}

func (fn *GenericMixedCmp) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{ \n"))
	fn.WriteBody(w)
	if sig.Err {
		w.Write([]byte("\nreturn\n"))
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

func makeGenericMixedCmps(tbo []TypedBinOp) (retVal []*GenericMixedCmp) {
	for _, tb := range tbo {
		if tc := tb.TypeClass(); tc != nil && !tc(tb.Kind()) {
			continue
		}
		fn := &GenericMixedCmp{
			GenericVecVecCmp: GenericVecVecCmp{
				TypedBinOp: tb,
			},
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
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
		g.RetSame = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
	}
}

func generateGenericMixedCmp(f io.Writer, ak Kinds) {
	gen := makeGenericMixedCmps(typedCmps)
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
		g.RetSame = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
		g.LeftVec = true
		g.RetSame = false
		g.Iter = false
	}

	// VS

	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
		g.RetSame = false
		g.Iter = true
	}
	for _, g := range gen {
		g.Write(f)
		g.RetSame = true
	}
	for _, g := range gen {
		if isBoolRepr(g.Kind()) {
			g.Write(f)
		}
	}
}

/* OTHER */

// element wise Min/Max
const genericElMinMaxRaw = `func VecMin{{short . | title}}(a, b []{{asType .}}) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv < v {
			a[i] = bv
		}
	}
	return nil
}
func VecMax{{short . | title}}(a, b []{{asType .}}) error {
	if len(a) != len(b) {
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)]
	for i, v := range a {
		bv := b[i]
		if bv > v {
			a[i] = bv
		}
	}
	return nil
}
`

// scalar Min/Max
const genericScalarMinMaxRaw = `func Min{{short .}}(a, b {{asType .}}) (c {{asType .}}) {if a < b {
	return a
	}
	return b
}

func Max{{short .}}(a, b {{asType .}}) (c {{asType .}}) {if a > b {
	return a
	}
	return b
}
`

var (
	genericElMinMax *template.Template
	genericMinMax   *template.Template
)

func init() {
	genericElMinMax = template.Must(template.New("genericVecVecMinMax").Funcs(funcs).Parse(genericElMinMaxRaw))
	genericMinMax = template.Must(template.New("genericMinMax").Funcs(funcs).Parse(genericScalarMinMaxRaw))
}

func generateMinMax(f io.Writer, ak Kinds) {
	for _, k := range filter(ak.Kinds, isOrd) {
		genericElMinMax.Execute(f, k)
	}

	for _, k := range filter(ak.Kinds, isOrd) {
		genericMinMax.Execute(f, k)
	}
}
