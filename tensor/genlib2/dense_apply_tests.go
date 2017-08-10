package main

import (
	"io"
	"text/template"
)

const testDenseApplyRaw = `func TestDense_Apply(t *testing.T){
{{range .Kinds -}}
{{if isParameterized . -}}
{{else -}}
	iden{{short .}} := func(a *QCDense{{short .}}) bool {
		var correct *Dense
		var ret Tensor
		var err error

		correct = newDense({{asType . | title | strip}}, a.len())
		correct.Memset({{if isNumber . -}}{{asType .}}(1))
		{{else if eq .String "bool" -}}true)
		{{else if eq .String "string" -}}"Hello World")
		{{else if eq .String "uintptr" -}}uintptr(0xdeadbeef))
		{{else if eq .String "unsafe.Pointer" -}}unsafe.Pointer(uintptr(0xdeadbeef)))
		{{end -}}

		if ret, err= a.Apply(mutate{{short .}}); err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()){
			t.Logf("ret.Data() %v || %v", ret.Data(), correct.Data())
			return false
		}

		// wrong function type
		if _, err = a.Apply({{if eq .String "float64"}}identityF32{{else}}identityF64{{end}}); err == nil {
			t.Error(err)
			return false
		}

		// sliced
		if a.len() > 10 {
			var b *Dense
			if b, err = sliceDense(a.Dense, makeRS(0, 10)); err != nil {
				t.Error(err)
				return false
			}
			if ret, err = b.Apply(mutate{{short .}}); err != nil {
				t.Error(err)
				return false
			}
			if !allClose(ret.Data(), correct.{{sliceOf .}}[0:10]){
				t.Logf("ret.Data() %v || %v", ret.Data(), correct.{{sliceOf .}}[0:10])
				return false
			}

			// wrong function type
			if _, err = b.Apply({{if eq .String "float64"}}identityF32{{else}}identityF64{{end}}); err == nil {
				t.Error(err)
				return false
			}
		}
		return true
	}
	if err := quick.Check(iden{{short . }}, nil); err != nil {
		t.Errorf("Applying mutation function to {{.}} failed: %v", err)
	}

	// safety:
	unsafe{{short .}} := func(a *QCDense{{short .}}) bool {
		var correct *Dense
		var ret Tensor
		var err error
		correct = newDense({{asType . | title | strip}}, a.len())
		copyDense(correct, a.Dense)

		// safe first
		if ret, err = a.Apply(identity{{short .}}); err != nil {
			t.Error(err)
			return false
		}
		if ret == a.Dense {
			t.Error("Expected ret != a")
			return false
		}

		// unsafe
		if ret, err = a.Apply(identity{{short .}}, UseUnsafe()); err != nil {
			t.Error(err)
			return false
		}
		if ret != a.Dense {
			t.Error("Expected ret == a")
			return false
		}
		return true
	}
	if err := quick.Check(unsafe{{short .}}, nil); err != nil {
		t.Errorf("Unsafe identity function for {{.}} failed %v", err)
	}

	{{if isNumber . -}}
	incr{{short .}} := func(a *QCDense{{short .}}) bool{
		var ret, correct Tensor
		var err error
		if correct, err = a.Add(a.Dense); err != nil {
			t.Error(err)
			return false
		}

		if ret, err = a.Apply(identity{{short .}}, WithIncr(a.Dense));err != nil {
			t.Error(err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()){
			return false
		}
		return true
	}
	if err := quick.Check(incr{{short . }}, nil); err != nil {
		t.Errorf("Applying identity function to {{.}} failed: %v", err)
	}

	{{end -}}
{{end -}}
{{end -}}
}
`

var (
	testDenseApply *template.Template
)

func init() {
	testDenseApply = template.Must(template.New("testDenseApply").Funcs(funcs).Parse(testDenseApplyRaw))
}

func generateDenseApplyTests(f io.Writer, generic Kinds) {
	testDenseApply.Execute(f, generic)
}
