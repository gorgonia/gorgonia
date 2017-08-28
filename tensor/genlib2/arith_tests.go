package main

import (
	"fmt"
	"io"
	"text/template"
)

const (
	APICallVVRaw = `ret, err := {{.Name}}(a, b {{template "funcoptuse"}})`
	APICallVSRaw = `ret, err := {{.Name}}(a, b {{template "funcoptuse"}})`
	APICallSVRaw = `ret, err := {{.Name}}(b, a {{template "funcoptuse"}})`

	APIInvVVRaw = `ret, err = {{.Inv}}(ret, b, UseUnsafe())`
	APIInvVSRaw = `ret, err = {{.Inv}}(ret, b, UseUnsafe())`
	APIInvSVRaw = `ret, err = {{.Name}}(b, ret, UseUnsafe())`

	DenseMethodCallVVRaw = `ret, err := a.{{.Name}}(b {{template "funcoptuse"}})`
	DenseMethodCallVSRaw = `ret, err := a.{{.Name}}Scalar(b, true {{template "funcoptuse"}})`
	DenseMethodCallSVRaw = `ret, err := a.{{.Name}}Scalar(b, false {{template "funcoptuse"}})`

	DenseMethodInvVVRaw = `ret, err = ret.{{.Inv}}(b, UseUnsafe())`
	DenseMethodInvVSRaw = `ret, err = ret.{{.Inv}}Scalar(b, true, UseUnsafe())`
	DenseMethodInvSVRaw = `ret, err = ret.{{.Name}}Scalar(b, false, UseUnsafe())`
)

type ArithTest struct {
	arithOp
	scalars bool
	funcOpt string
	lvl     Level
}

func (fn *ArithTest) Signature() *Signature {
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
	if fn.funcOpt != "" {
		name += "_" + fn.funcOpt
	}
	return &Signature{
		Name:           name,
		NameTemplate:   plainName,
		ParamNames:     []string{"t"},
		ParamTemplates: []*template.Template{testingType},
	}
}

func (fn *ArithTest) WriteBody(w io.Writer) {
	if fn.HasIdentity {
		fn.writeIdentity(w)
		fmt.Fprintf(w, "\n")
	}

	if fn.IsInv {
		fn.writeInv(w)
	}
}

func (fn *ArithTest) canWrite() bool {
	if fn.HasIdentity || fn.IsInv {
		return true
	}
	return false
}

func (fn *ArithTest) writeIdentity(w io.Writer) {
	var t *template.Template
	if fn.scalars {
		t = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithScalarTestRaw))
	} else {
		t = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithTestBodyRaw))
	}
	switch fn.lvl {
	case API:
		if fn.scalars {
			template.Must(t.New("call0").Parse(APICallVSRaw))
			template.Must(t.New("call1").Parse(APICallSVRaw))
		} else {
			template.Must(t.New("call0").Parse(APICallVVRaw))
		}
	case Dense:
		if fn.scalars {
			template.Must(t.New("call0").Parse(DenseMethodCallVSRaw))
			template.Must(t.New("call1").Parse(DenseMethodCallSVRaw))
		} else {
			template.Must(t.New("call0").Parse(DenseMethodCallVVRaw))
		}

	}
	template.Must(t.New("funcoptdecl").Parse(funcOptDecl[fn.funcOpt]))
	template.Must(t.New("funcoptcorrect").Parse(funcOptCorrect[fn.funcOpt]))
	template.Must(t.New("funcoptuse").Parse(funcOptUse[fn.funcOpt]))
	template.Must(t.New("funcoptcheck").Parse(funcOptCheck[fn.funcOpt]))

	t.Execute(w, fn)
}

func (fn *ArithTest) writeInv(w io.Writer) {
	var t *template.Template
	if fn.scalars {
		t = template.Must(template.New("dense involution test").Funcs(funcs).Parse(denseInvArithScalarTestRaw))
	} else {
		t = template.Must(template.New("dense involution test").Funcs(funcs).Parse(denseInvArithTestBodyRaw))
	}
	switch fn.lvl {
	case API:
		if fn.scalars {
			template.Must(t.New("call0").Parse(APICallVSRaw))
			template.Must(t.New("call1").Parse(APICallSVRaw))
			template.Must(t.New("callInv0").Parse(APIInvVSRaw))
			template.Must(t.New("callInv1").Parse(APIInvSVRaw))
		} else {
			template.Must(t.New("call0").Parse(APICallVVRaw))
			template.Must(t.New("callInv").Parse(APIInvVVRaw))
		}
	case Dense:
		if fn.scalars {
			template.Must(t.New("call0").Parse(DenseMethodCallVSRaw))
			template.Must(t.New("call1").Parse(DenseMethodCallSVRaw))
			template.Must(t.New("callInv0").Parse(DenseMethodInvVSRaw))
			template.Must(t.New("callInv1").Parse(DenseMethodInvSVRaw))
		} else {
			template.Must(t.New("call0").Parse(DenseMethodCallVVRaw))
			template.Must(t.New("callInv").Parse(DenseMethodInvVVRaw))
		}
	}

	template.Must(t.New("funcoptdecl").Parse(funcOptDecl[fn.funcOpt]))
	template.Must(t.New("funcoptcorrect").Parse(funcOptCorrect[fn.funcOpt]))
	template.Must(t.New("funcoptuse").Parse(funcOptUse[fn.funcOpt]))
	template.Must(t.New("funcoptcheck").Parse(funcOptCheck[fn.funcOpt]))

	t.Execute(w, fn)
}

func (fn *ArithTest) Write(w io.Writer) {
	sig := fn.Signature()
	w.Write([]byte("func "))
	sig.Write(w)
	w.Write([]byte("{\nvar r *rand.Rand\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func generateAPIArithTests(f io.Writer, ak Kinds) {
	var tests []*ArithTest
	for _, op := range arithBinOps {
		t := &ArithTest{
			arithOp: op,
			lvl:     API,
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
	}
}

func generateAPIArithScalarTests(f io.Writer, ak Kinds) {
	var tests []*ArithTest
	for _, op := range arithBinOps {
		t := &ArithTest{
			arithOp: op,
			scalars: true,
			lvl:     API,
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
	}
}

func generateDenseMethodArithTests(f io.Writer, ak Kinds) {
	var tests []*ArithTest
	for _, op := range arithBinOps {
		t := &ArithTest{
			arithOp: op,
			lvl:     Dense,
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
	}
}

func generateDenseMethodScalarTests(f io.Writer, ak Kinds) {
	var tests []*ArithTest
	for _, op := range arithBinOps {
		t := &ArithTest{
			arithOp: op,
			scalars: true,
			lvl:     Dense,
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.canWrite() {
			fn.Write(f)
		}
	}
}
