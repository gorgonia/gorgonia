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

if {{lower .Name}}er, ok := e.({{if $eleqne }}ElEqer{{else}}{{.Name}}{{end}}er); ok {
	var ret Tensor
	if ret, err = {{lower .Name}}er.{{.Name}}(t, other, opts...); err != nil {
		err = errors.Wrapf(err, "Unable to do {{.Name}}()")
		return
	}
	if retVal, ok = ret.(*Dense); !ok {
		err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
	}
	return
}
return  nil, errors.Errorf("Engine does not support {{.Name}}()")
`

const denseArithScalarBodyRaw = `e := t.e
	if e == nil {
		e = StdEng{}
	}

{{$elne := eq .Name "Ne"}}
{{$eleq := eq .Name "Eq"}}
{{$eleqne := or $eleq $elne}}
	if {{lower .Name}}er, ok := e.({{if $eleqne }}El{{end}}{{.Name}}er); ok {
		var ret Tensor
		if ret, err = {{lower .Name}}er.{{.Name}}Scalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do {{.Name}}Scalar()")
			return
		}
		if retVal, ok = ret.(*Dense); ok {
			err = errors.Errorf("Expected a *Dense. Got %T instead", ret)
		}
		return
	}
	return nil, errors.Errorf("Engine does not support {{.Name}}()")
`

var (
	denseArithBody       *template.Template
	denseArithScalarBody *template.Template
)

func init() {
	denseArithBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithBodyRaw))
	denseArithScalarBody = template.Must(template.New("dense arith body").Funcs(funcs).Parse(denseArithScalarBodyRaw))
}
