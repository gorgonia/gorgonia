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

const denseIdentityArithTestBodyRaw = `r = rand.New(rand.NewSource(time.Now().UnixNano()))
iden := func(a *Dense) bool {
	b := New(Of(a.t), WithShape(a.Shape().Clone()...))
	{{if ne .Identity 0 -}}
			b.Memset(identityVal({{.Identity}}, a.t))
	{{end -}}
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	
	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("a.Dtype: %v", a.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
		if err := quick.Check(iden, &quick.Config{Rand: r}); err != nil{
			t.Errorf("Identity test for {{.Name}} failed: %v", err)
		}
`

const denseIdentityArithScalarTestRaw = `r = rand.New(rand.NewSource(time.Now().UnixNano()))
iden1 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}

if err := quick.Check(iden1, &quick.Config{Rand:r}); err != nil {
	t.Errorf("Identity test for {{.Name}} (tensor as left, scalar as right) failed: %v", err)
}

{{if .IsCommutative -}}

iden2 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call1" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}

	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
r = rand.New(rand.NewSource(time.Now().UnixNano()))
if err := quick.Check(iden2, &quick.Config{Rand:r}); err != nil {
	t.Errorf("Identity test for {{.Name}} (scalar as left, tensor as right) failed: %v", err)
}
{{end -}}
`

const denseInvArithTestBodyRaw = `r = rand.New(rand.NewSource(time.Now().UnixNano()))
inv := func(a *Dense) bool {
	b := New(Of(a.t), WithShape(a.Shape().Clone()...))
	{{if ne .Identity 0 -}}
			b.Memset(identityVal({{.Identity}}, a.t))
	{{end -}}
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv" .}}
	
	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("a.Dtype: %v", a.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
		if err := quick.Check(inv, &quick.Config{Rand:r}); err != nil{
			t.Errorf("Inv test for {{.Name}} failed: %v", err)
		}
`

const denseInvArithScalarTestRaw = `r = rand.New(rand.NewSource(time.Now().UnixNano()))
inv1 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl"}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call0" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}VS", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv0" .}}

	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("Correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
if err := quick.Check(inv1, &quick.Config{Rand:r}); err != nil {
	t.Errorf("Inv test for {{.Name}} (tensor as left, scalar as right) failed: %v", err)
}

{{if .IsInvolutionary -}}
inv2 := func(q *Dense) bool {
	a := q.Clone().(*Dense)
	b := identityVal({{.Identity}}, q.t)
	{{template "funcoptdecl" -}}
	correct := a.Clone().(*Dense)
	{{template "funcoptcorrect" -}}

	we := willerr(a, {{.TypeClassName}})
	{{template "call1" . }}
	if err, retEarly := qcErrCheck(t, "{{.Name}}SV", a, b, we, err); retEarly{
		if err != nil {
			return false
		}
		return true
	}
	{{template "callInv1" .}}

	isFloatTypes := qcIsFloat(a)
	if (isFloatTypes && !allClose(correct.Data(), ret.Data())) || (!isFloatTypes && !reflect.DeepEqual(correct.Data(), ret.Data())) {
		t.Errorf("q.Dtype: %v", q.Dtype())
		t.Errorf("correct.Data()\n%v", correct.Data())
		t.Errorf("ret.Data()\n%v", ret.Data())
		return false
	}
	{{template "funcoptcheck" -}}

	return true
}
r = rand.New(rand.NewSource(time.Now().UnixNano()))
if err := quick.Check(inv2, &quick.Config{Rand:r}); err != nil {
	t.Errorf("Inv test for {{.Name}} (scalar as left, tensor as right) failed: %v", err)
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
