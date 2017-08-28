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

const denseIdentityArithTestBodyRaw = `iden := func(a *Dense) bool {
	identity := New(Of(a.t), WithShape(a.Shape().Clone()...))
	{{if ne .Identity 0 -}}
			identity.Memset(identityVal({{.Identity}}, a.t))
	{{end -}}
	{{template "funcoptdecl"}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect"}}

	we := willerr(a, {{.TypeClassName}})
	ret, err := {{.Name}}(a, identity {{template "funcoptuse"}})
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, identity, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	
	var isFloatTypes bool
	if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
		isFloatTypes = true
	}
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("a.Dtype: %v", a.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	return true
}
		if err := quick.Check(iden, nil); err != nil{
			t.Errorf("Identity test for {{.Name}} failed: %v", err)
		}
`

const denseIdentityArithScalarTestRaw = `iden1 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	identity := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
	copyDense(correct, a)
	{{template "funcoptcorrect"}}

	we := willerr(a, {{.TypeClassName}})
	ret, err := {{.Name}}(a, identity {{template "funcoptuse"}})
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, identity, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
	}
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	return true
}
if err := quick.Check(iden1, nil); err != nil {
	t.Errorf("Identity test for {{.Name}} (tensor as left, scalar as right) failed: %v", err)
}

{{if .IsCommutative -}}

iden2 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	identity := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := New(Of(a.t), WithShape(a.Shape().Clone()...))
	copyDense(correct, a)
	{{template "funcoptcorrect"}}

	we := willerr(a, {{.TypeClassName}})
	ret, err := {{.Name}}( identity, a {{template "funcoptuse"}})
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, identity, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	var isFloatTypes bool
		if err := typeclassCheck(a.Dtype(), floatcmplxTypes); err == nil {
			isFloatTypes = true
	}
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck"}}

	return true
}
if err := quick.Check(iden2, nil); err != nil {
	t.Errorf("Identity test for {{.Name}} (scalar as left, tensor as right) failed: %v", err)
}
{{end -}}
`

var (
	denseArithBody       *template.Template
	denseArithScalarBody *template.Template

	denseIdentityArithTest       *template.Template
	denseIdentityArithScalarTest *template.Template
)

func init() {
	denseArithBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithBodyRaw))
	denseArithScalarBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithScalarBodyRaw))

	denseIdentityArithTest = template.Must(template.New("dense identity test").Funcs(funcs).Parse(denseIdentityArithTestBodyRaw))
	denseIdentityArithScalarTest = template.Must(template.New("dense scalar identity test").Funcs(funcs).Parse(denseIdentityArithScalarTestRaw))
}
