package main

import (
	"fmt"
	"io"
	"text/template"
)

const (
	APICallVVaxbRaw    = `axb, err := {{.Name}}(a, b {{template "funcoptuse" . -}})`
	APICallVVbxcRaw    = `bxc, err := {{.Name}}(b, c {{template "funcoptuse" . -}})`
	APICallVVaxcRaw    = `axc, err := {{.Name}}(a, c {{template "funcoptuse" . -}})`
	APICallMixedaxbRaw = `axb, err := {{.Name}}(a, b {{template "funcoptuse" . -}})`
	APICallMixedbxcRaw = `bxc, err := {{.Name}}(b, c {{template "funcoptuse" . -}})`
	APICallMixedaxcRaw = `axc, err := {{.Name}}(a, c {{template "funcoptuse" . -}})`

	DenseMethodCallVVaxbRaw    = `axb, err := a.{{.Name}}(b {{template "funcoptuse" . -}})`
	DenseMethodCallVVbxcRaw    = `bxc, err := b.{{.Name}}(c {{template "funcoptuse" . -}})`
	DenseMethodCallVVaxcRaw    = `axc, err := a.{{.Name}}(c {{template "funcoptuse" . -}})`
	DenseMethodCallMixedaxbRaw = `axb, err := a.{{.Name}}Scalar(b, true {{template "funcoptuse" . -}})`
	DenseMethodCallMixedbxcRaw = `bxc, err := c.{{.Name}}Scalar(b, false {{template "funcoptuse" . -}})`
	DenseMethodCallMixedaxcRaw = `axc, err := a.{{.Name}}(c, true {{template "funcoptuse" . -}})`
)

const transitivityBodyRaw = `transFn := func(q *Dense) bool {
	we, willFailEq := willerr(q, {{.TypeClassName}}, {{.EqFailTypeClassName}})

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	a := q.Clone().(*Dense)
	b := q.Clone().(*Dense)
	c := q.Clone().(*Dense)

	bv, _ := quick.Value(b.Dtype().Type, r)
	cv, _ := quick.Value(c.Dtype().Type, r)
	b.Memset(bv.Interface())
	c.Memset(cv.Interface())

	{{template "axb" .}}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙b", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	{{template "bxc" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - b∙c", b, c, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	{{template "axc" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙c", a, c, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	ab := axb.(*Dense).Bools()
	bc := bxc.(*Dense).Bools()
	ac := axc.(*Dense).Bools()

	for i, vab := range ab {
		if vab && bc[i] {
			if !ac[i]{
				return false
			}
		}
	}
	return true
}
r = rand.New(rand.NewSource(time.Now().UnixNano()))
if err := quick.Check(transFn, &quick.Config{Rand: r}); err != nil {
	t.Error("Transitivity test for {{.Name}} failed: %v", err)
}
`

const transitivityMixedBodyRaw = `transFn := func(q *Dense) bool {
	we, willFailEq := willerr(q, {{.TypeClassName}}, {{.EqFailTypeClassName}})	

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	a := q.Clone().(*Dense)
	bv, _ := quick.Value(a.Dtype().Type, r)
	b := bv.Interface()
	c := q.Clone().(*Dense)
	cv, _ := quick.Value(c.Dtype().Type, r)
	c.Memset(cv.Interface())

	{{template "axb" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙b", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	{{template "bxc" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - b∙c", b, c, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	{{template "axc" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}} - a∙c", a, c, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}











	ab := axb.(*Dense).Bools()
	bc := bxc.(*Dense).Bools()
	ac := axc.(*Dense).Bools()

	for i, vab := range ab {
		if vab && bc[i] {
			if !ac[i]{
				return false
			}
		}
	}
	return true
}

`

type CmpTest struct {
	cmpOp
	scalars             bool
	lvl                 Level
	FuncOpt             string
	EqFailTypeClassName string
}

func (fn *CmpTest) Signature() *Signature {
	var name string
	switch fn.lvl {
	case API:
		name = fmt.Sprintf("Test%s", fn.Name())
	case Dense:
		name = fmt.Sprintf("TestDense_%s", fn.Name())
	}
	if fn.scalars {
		name += "Scalar"
	}
	if fn.FuncOpt != "" {
		name += "_" + fn.FuncOpt
	}
	return &Signature{
		Name:           name,
		NameTemplate:   plainName,
		ParamNames:     []string{"t"},
		ParamTemplates: []*template.Template{testingType},
	}
}

func (fn *CmpTest) WriteBody(w io.Writer) {
	if fn.IsTransitive {
		fn.writeTransitivity(w)
	}
}

func (fn *CmpTest) writeTransitivity(w io.Writer) {
	var t *template.Template
	if fn.scalars {
		t = template.Must(template.New("dense cmp transitivity test").Funcs(funcs).Parse(transitivityMixedBodyRaw))
	} else {
		t = template.Must(template.New("dense cmp transitivity test").Funcs(funcs).Parse(transitivityBodyRaw))
	}

	switch fn.lvl {
	case API:
		if fn.scalars {
			template.Must(t.New("axb").Parse(APICallMixedaxbRaw))
			template.Must(t.New("bxc").Parse(APICallMixedbxcRaw))
			template.Must(t.New("axc").Parse(APICallMixedaxcRaw))
		} else {
			template.Must(t.New("axb").Parse(APICallVVaxbRaw))
			template.Must(t.New("bxc").Parse(APICallVVbxcRaw))
			template.Must(t.New("axc").Parse(APICallVVaxcRaw))
		}
	case Dense:
		if fn.scalars {
			template.Must(t.New("axb").Parse(DenseMethodCallMixedaxbRaw))
			template.Must(t.New("bxc").Parse(DenseMethodCallMixedbxcRaw))
			template.Must(t.New("axc").Parse(DenseMethodCallMixedaxcRaw))
		} else {
			template.Must(t.New("axb").Parse(DenseMethodCallVVaxbRaw))
			template.Must(t.New("bxc").Parse(DenseMethodCallVVbxcRaw))
			template.Must(t.New("axc").Parse(DenseMethodCallVVaxcRaw))
		}
	}

	template.Must(t.New("funcoptdecl").Parse(funcOptDecl[fn.FuncOpt]))
	template.Must(t.New("funcoptcorrect").Parse(funcOptCorrect[fn.FuncOpt]))
	template.Must(t.New("funcoptuse").Parse(funcOptUse[fn.FuncOpt]))
	template.Must(t.New("funcoptcheck").Parse(funcOptCheck[fn.FuncOpt]))

	t.Execute(w, fn)
}

func (fn *CmpTest) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\nvar r *rand.Rand\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func generateAPICmpTests(f io.Writer, ak Kinds) {
	var tests []*CmpTest

	for _, op := range cmpBinOps {
		t := &CmpTest{
			cmpOp:               op,
			lvl:                 API,
			EqFailTypeClassName: "nil",
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		fn.Write(f)
	}
}
