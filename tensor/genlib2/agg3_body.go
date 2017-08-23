package main

import "text/template"

// 3rd level function aggregation templates

const denseArithBodyRaw = `e := t.e
if e == nil {
	e = StdEng{}
}
{{$elne := eq .Name "Ne"}}
{{$eleq := eq .Name "Eq"}}
{{$eleqne := or $eleq $elne}}

if {{interfaceName .Name | lower}}, ok := e.({{interfaceName .Name}}); ok {
	var ret Tensor
	if ret, err = {{interfaceName .Name | lower}}.{{if $eleqne}}El{{end}}{{.Name}}(t, other, opts...); err != nil {
		err = errors.Wrapf(err, "Unable to do {{.Name}}()")
		return
	}
	if retVal, err = assertDense(ret); err != nil {
		return nil, errors.Wrapf(err, opFail, "{{.Name}}")
	}
	return
}
return  nil, errors.Errorf("Engine does not support {{.Name}}()")
`

const denseArithScalarBodyRaw = `e := t.e
	if e == nil {
		e = StdEng{}
	}

	if {{interfaceName .Name | lower}}, ok := e.({{interfaceName .Name}}); ok {
		var ret Tensor
		if ret, err = {{interfaceName .Name | lower}}.{{.Name}}Scalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do {{.Name}}Scalar()")
			return
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "{{.Name}}Scalar")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support {{.Name}}Scalar()")
`

const denseIdentityArithTestBodyRaw = `iden := func(a *QCDenseF64) bool {
	identity := New(Of(Float64), WithShape(a.len()))
	{{if ne .Identity 0 -}}
			identity.Memset({{.Identity}}.0)
	{{end -}}
	{{template "funcoptdecl"}}
	correct := New(Of(Float64), WithShape(a.len()))
	copyDense(correct, a)
	{{template "funcoptcorrect"}}

	ret, err := {{.Name}}(a, identity {{template "funcoptuse"}})
	if err != nil {
			t.Errorf("Identity tests for {{.Name}} was unable to proceed: %v", err)
			return false
	}
	if !allClose(correct.Data(), ret.Data()) {
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	return true
}
		if err := quick.Check(iden, nil); err != nil{
			t.Errorf("Identity test for {{.Name}} failed: %v", err)
		}

	idenSliced := func(a *QCDenseF64) bool {
		a1, _ := sliceDense(a.Dense, makeRS(0,5))
		identity := New(Of(Float64), WithShape(a.len()))
		{{if ne .Identity 0 -}}
				identity.Memset({{.Identity}}.0)
		{{end -}}
		{{template "funcoptdecl"}}
		correct := New(Of(Float64), WithShape(5))
		copyDense(correct, a1)
		{{template "funcoptcorrect"}}

		ret, err := {{.Name}}(a, identity {{template "funcoptuse"}})
		if err != nil {
			t.Errorf("Identity sliced test for {{.Name}} was unable to proceed: %v", err)
			return false
		}
		if !allClose(correct.Data(), ret.Data()) {
			t.Errorf("Correct.Data()\n%v", correct.Data())
				t.Errorf("ret.Data()\n%v", ret.Data())
			return false
		}
		{{template "funcoptcheck"}}
		return true

	}

	if err := quick.Check(idenSliced, nil); err != nil{
			t.Errorf("IdentitySliced test for {{.Name}} failed: %v", err)
	}
`

const denseIdentityArithScalarTestBodyRaw = `iden := func(q *QCDenseF64) bool {
	a := &QCDenseF64{q.Dense.Clone().(*Dense)}
	identity := {{.Identity}}.0
	{{template "funcoptdecl"}}
	correct := New(Of(Float64), WithShape(a.len()))
	copyDense(correct, a)
	{{template "funcoptcorrect"}}

	ret, err := {{.Name}}Scalar(a, identity, true, {{template "funcoptuse"}})
	if err != nil {
		t.Errorf("Identity tests for {{.Name}} the tensor in left operand was unable to proceed: %v", err)
	}

	if !allClose(correct.Data(), ret.Data()) {
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	a = &QCDenseF64{q.Dense.Clone().(*Dense)}
	identity := {{.Identity}}.0
	{{template "funcoptdecl"}}
	correct := New(Of(Float64), WithShape(a.len()))
	copyDense(correct, a)
	{{template "funcoptcorrect"}}

	ret, err := {{.Name}}Scalar(a, identity, false, {{template "funcoptuse"}})
	if err != nil {
		t.Errorf("Identity tests for {{.Name}} the tensor in left operand was unable to proceed: %v", err)
	}

	if !allClose(correct.Data(), ret.Data()) {
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	return true
}

if err := quick.Check(iden, nil); err != nil {
	t.Errorf("Identity test for {{.Name}}Scalar failed: %v", err)
}

`

var (
	denseArithBody       *template.Template
	denseArithScalarBody *template.Template

	denseIdentityArithTest *template.Template
)

func init() {
	denseArithBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithBodyRaw))
	denseArithScalarBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithScalarBodyRaw))

	denseIdentityArithTest = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithTestBodyRaw))
}
