package main

import (
	"fmt"
	"io"
	"text/template"
)

const unaryTestBodyRaw = `invFn := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	ret, err := {{.Name}}(a {{template "funcoptuse"}})
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, nil, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{if ne .InvTypeClass "" -}}
	if err := typeclassCheck(a.Dtype(), {{.InvTypeClass}}); err != nil {
		return true // uninvertible due to type class implementation issues
	}
	{{end -}}
	{{.Inv}}(ret, UseUnsafe())
	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}
	return true
}

r = rand.New(rand.NewSource(time.Now().UnixNano()))
if err := quick.Check(invFn, &quick.Config{Rand:r}); err != nil{
	t.Errorf("Inv tests for {{.Name}} failed: %v", err)
}
`

type unaryTest struct {
	unaryOp
	FuncOpt             string
	EqFailTypeClassName string
	InvTypeClass        string
}

func (fn *unaryTest) Signature() *Signature {
	name := fmt.Sprintf("Test%s", fn.Name())
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

func (fn *unaryTest) WriteBody(w io.Writer) {
	t := template.Must(template.New("unary test body").Funcs(funcs).Parse(unaryTestBodyRaw))
	template.Must(t.New("funcoptdecl").Parse(funcOptDecl[fn.FuncOpt]))
	template.Must(t.New("funcoptcorrect").Parse(funcOptCorrect[fn.FuncOpt]))
	template.Must(t.New("funcoptuse").Parse(funcOptUse[fn.FuncOpt]))
	template.Must(t.New("funcoptcheck").Parse(funcOptCheck[fn.FuncOpt]))
	t.Execute(w, fn)
}

func (fn *unaryTest) canWrite() bool { return fn.Inv != "" }

func (fn *unaryTest) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\nvar r *rand.Rand\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func generateAPIUnaryTests(f io.Writer, ak Kinds) {
	var tests []*unaryTest
	for _, op := range conditionalUnaries {
		t := &unaryTest{
			unaryOp:             op,
			EqFailTypeClassName: "nil",
		}

		tests = append(tests, t)
	}

	for _, op := range unconditionalUnaries {
		t := &unaryTest{
			unaryOp:             op,
			EqFailTypeClassName: "nil",
		}
		switch op.name {
		case "Square":
			t.InvTypeClass = "floatcmplxTypes"
		case "Cube":
			t.InvTypeClass = "floatTypes"
		}

		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.FuncOpt = "incr"
	}

	// for now incr cannot be quickchecked

	// for _, fn := range tests {
	// 	if fn.canWrite() {
	// 		fn.Write(f)
	// 	}
	// }
}
