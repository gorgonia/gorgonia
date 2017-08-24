package main

import (
	"fmt"
	"io"
	"text/template"
)

type ArithTest struct {
	arithOp
	scalars bool
	funcOpt string
}

func (fn *ArithTest) Signature() *Signature {
	name := fmt.Sprintf("Test%s", fn.Name())
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
	var t *template.Template
	if fn.scalars {
		t = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithScalarTestRaw))
	} else {
		t = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithTestBodyRaw))
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
	w.Write([]byte("{\n"))
	fn.WriteBody(w)
	w.Write([]byte("}\n"))
}

func generateAPIArithTests(f io.Writer, ak Kinds) {
	var tests []*ArithTest
	for _, op := range arithBinOps {
		t := &ArithTest{
			arithOp: op,
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests { 
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.HasIdentity {
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
		}
		tests = append(tests, t)
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "unsafe"
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "reuse"
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
		fn.funcOpt = "incr"
	}

	for _, fn := range tests {
		if fn.HasIdentity {
			fn.Write(f)
		}
	}
}
