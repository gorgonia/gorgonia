package main

import "text/template"

// 3rd level function aggregation templates

const denseArithBodyRaw = `{{$elne := eq .Name "Ne"}}
{{$eleq := eq .Name "Eq"}}
{{$eleqne := or $eleq $elne}}
var ret Tensor
if t.oe != nil {
	if ret, err = t.oe.{{if $eleqne}}El{{end}}{{.Name}}(t, other, opts...); err != nil {
		return nil, errors.Wrapf(err, "Unable to do {{.Name}}()")
	}
	if retVal, err = assertDense(ret); err != nil {
		return nil, errors.Wrapf(err, opFail, "{{.Name}}")
	}
	return
}

if {{interfaceName .Name | lower}}, ok := t.e.({{interfaceName .Name}}); ok {
	if ret, err = {{interfaceName .Name | lower}}.{{if $eleqne}}El{{end}}{{.Name}}(t, other, opts...); err != nil {
		return nil, errors.Wrapf(err, "Unable to do {{.Name}}()")
	}
	if retVal, err = assertDense(ret); err != nil {
		return nil, errors.Wrapf(err, opFail, "{{.Name}}")
	}
	return
}
return  nil, errors.Errorf("Engine does not support {{.Name}}()")
`

const denseArithScalarBodyRaw = `var ret Tensor
if t.oe != nil {
	if ret, err = t.oe.{{.Name}}Scalar(t, other, leftTensor, opts...); err != nil{
		return nil, errors.Wrapf(err, "Unable to do {{.Name}}Scalar()")
	}
	if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "{{.Name}}Scalar")
		}
		return
}

	if {{interfaceName .Name | lower}}, ok := t.e.({{interfaceName .Name}}); ok {
		if ret, err = {{interfaceName .Name | lower}}.{{.Name}}Scalar(t, other, leftTensor, opts...); err != nil {
			return nil, errors.Wrapf(err, "Unable to do {{.Name}}Scalar()")
		}
		if retVal, err = assertDense(ret); err != nil {
			return nil, errors.Wrapf(err, opFail, "{{.Name}}Scalar")
		}
		return
	}
	return nil, errors.Errorf("Engine does not support {{.Name}}Scalar()")
`

const denseIdentityArithTestBodyRaw = `iden := func(a *Dense) bool {
	b := New(Of(a.t), WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
	{{if ne .Identity 0 -}}
			b.Memset(identityVal({{.Identity}}, a.t))
	{{end -}}
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := a.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	
	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
		if err := quick.Check(iden, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil{
			t.Errorf("Identity test for {{.Name}} failed: %v", err)
		}
`

const denseIdentityArithScalarTestRaw = `iden1 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := q.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}

if err := quick.Check(iden1, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
	t.Errorf("Identity test for {{.Name}} (tensor as left, scalar as right) failed: %v", err)
}

{{if .IsCommutative -}}
iden2 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := q.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call1" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
if err := quick.Check(iden2, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
	t.Errorf("Identity test for {{.Name}} (scalar as left, tensor as right) failed: %v", err)
}
{{end -}}
`

const denseInvArithTestBodyRaw = `inv := func(a *Dense) bool {
	b := New(Of(a.t), WithShape(a.Shape().Clone()...), WithEngine(a.Engine()))
	{{if ne .Identity 0 -}}
			b.Memset(identityVal({{.Identity}}, a.t))
	{{end -}}
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a,  {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := a.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv" .}}
	
	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
		if err := quick.Check(inv, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil{
			t.Errorf("Inv test for {{.Name}} failed: %v", err)
		}
`

const denseInvArithScalarTestRaw = `inv1 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := q.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}VS", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv0" .}}

	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
if err := quick.Check(inv1, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
	t.Errorf("Inv test for {{.Name}} (tensor as left, scalar as right) failed: %v", err)
}

{{if .IsInvolutionary -}}
{{if eq .FuncOpt "incr" -}}
{{else -}}
inv2 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we, willFailEq := willerr(a, {{.TypeClassName}}, {{.EqFailTypeClassName}})
	_, ok := q.Engine().({{interfaceName .Name}}); we = we || !ok

	{{template "call1" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}SV", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv1" .}}

	if !qcEqCheck(t, a.Dtype(), willFailEq, correct.Data(), ret.Data()) {
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
if err := quick.Check(inv2, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
	t.Errorf("Inv test for {{.Name}} (scalar as left, tensor as right) failed: %v", err)
}
{{end -}}
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
