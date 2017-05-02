package main

import (
	"io"
	"text/template"
)

const clampTestsRaw = `var clampTests = []struct {
	a, reuse interface{}
	min, max interface{}
	correct interface{}
}{
	{{range .Kinds -}}
	{{if isOrd . -}}
	{{if isNumber . -}}
	{[]{{asType .}}{1,2,3,4}, []{{asType .}}{10, 20, 30, 40}, {{asType .}}(2), {{asType .}}(3), []{{asType .}}{2,2,3,3}},
	{{end -}}
	{{end -}}
	{{end -}}
}

var clampTestsMasked = []struct {
	a, reuse interface{}
	min, max interface{}
	correct interface{}
}{
	{{range .Kinds -}}
	{{if isOrd . -}}
	{{if isNumber . -}}
	{[]{{asType .}}{1,2,3,4}, []{{asType .}}{1, 20, 30, 40}, {{asType .}}(2), {{asType .}}(3), []{{asType .}}{1,2,3,3}},
	{{end -}}
	{{end -}}
	{{end -}}
}

func TestClamp(t *testing.T){
	assert := assert.New(t)
	var got Tensor
	var T, reuse *Dense
	var err error
	for _, ct := range clampTests{
		T = New(WithBacking(ct.a))
		// safe
		if got, err = Clamp(T, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}
		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// reuse
		reuse = New(WithBacking(ct.reuse))
		if got, err = Clamp(T, ct.min, ct.max, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}
		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// unsafe
		if got, err = Clamp(T, ct.min, ct.max, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(ct.correct, got.Data())
	}
}

func TestClampMasked(t *testing.T){
	assert := assert.New(t)
	var got Tensor
	var T, reuse *Dense
	var err error
	for _, ct := range clampTestsMasked{
		T = New(WithBacking(ct.a, []bool{true,false,false,false}))
		// safe
		if got, err = Clamp(T, ct.min, ct.max); err != nil {
			t.Error(err)
			continue
		}
		if got == T {
			t.Error("expected got != T")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// reuse
		reuse = New(WithBacking(ct.reuse, []bool{true,false,false,false}))
		if got, err = Clamp(T, ct.min, ct.max, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}
		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(ct.correct, got.Data())

		// unsafe
		if got, err = Clamp(T, ct.min, ct.max, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(ct.correct, got.Data())
	}
}
`

const signTestsRaw = `var signTests = []struct {
	a, reuse interface{}
	correct interface{}
}{
	{{range .Kinds -}}
	{{$isInt := hasPrefix .String "int" -}}
	{{$isFloat := hasPrefix .String "float" -}}
	{{if isOrd . -}}
	{{if or $isFloat $isInt -}}
	{[]{{asType .}}{1,2,-2,-1}, []{{asType .}}{10, 20, 30, 40}, []{{asType .}}{1,1,-1,-1}},
	{{end -}}
	{{end -}}
	{{end -}}
}

var signTestsMasked = []struct {
	a, reuse interface{}
	correct interface{}
}{
	{{range .Kinds -}}
	{{$isInt := hasPrefix .String "int" -}}
	{{$isFloat := hasPrefix .String "float" -}}
	{{if isOrd . -}}
	{{if or $isFloat $isInt -}}
	{[]{{asType .}}{1,2,-2,-1}, []{{asType .}}{10, 20, 30, 40}, []{{asType .}}{1,1,-2,-1}},
	{{end -}}
	{{end -}}
	{{end -}}
}

func TestSign(t *testing.T){
	assert := assert.New(t)
	var  got Tensor
	var T, reuse *Dense
	var err error
	for _, st := range signTests{
		T = New(WithBacking(st.a))
		// safe
		if got, err = Sign(T); err != nil {
			t.Error(err)
			continue
		}

		if got == T {
			t.Error("expected got != T")
			continue	
		}
		assert.Equal(st.correct, got.Data())

		// reuse
		reuse = New(WithBacking(st.reuse))
		if got, err = Sign(T, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}

		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// unsafe
		if got, err = Sign(T, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(st.correct, got.Data())
	}
}

func TestSignMasked(t *testing.T){
	assert := assert.New(t)
	var  got Tensor
	var T, reuse *Dense
	var err error
	for _, st := range signTestsMasked{
		T = New(WithBacking(st.a,[]bool{false,false,true,false}))
		// safe
		if got, err = Sign(T); err != nil {
			t.Error(err)
			continue
		}

		if got == T {
			t.Error("expected got != T")
			continue	
		}
		assert.Equal(st.correct, got.Data())

		// reuse
		reuse = New(WithBacking(st.reuse,[]bool{false,false,true,false}))
		if got, err = Sign(T, WithReuse(reuse)); err != nil {
			t.Error(err)
			continue
		}

		if got != reuse {
			t.Error("expected got == reuse")
			continue
		}
		assert.Equal(st.correct, got.Data())

		// unsafe
		if got, err = Sign(T, UseUnsafe()); err != nil {
			t.Error(err)
			continue
		}
		if got != T {
			t.Error("expected got == T")
			continue
		}
		assert.Equal(st.correct, got.Data())
	}
}
`

var (
	clampTest *template.Template
	signTest  *template.Template
)

func init() {
	clampTest = template.Must(template.New("clampTest").Funcs(funcs).Parse(clampTestsRaw))
	signTest = template.Must(template.New("signTest").Funcs(funcs).Parse(signTestsRaw))
}

func generateUnaryTests(f io.Writer, generic *ManyKinds) {
	clampTest.Execute(f, generic)
	signTest.Execute(f, generic)
}
