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
	if retVal, ok = ret.(*Dense); !ok {
		err = errors.Errorf("Unable to do {{.Name}} - Expected a %T. Got %T instead", retVal, ret)
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
	if {{interfaceName .Name | lower}}, ok := e.({{interfaceName .Name}}); ok {
		var ret Tensor
		if ret, err = {{interfaceName .Name | lower}}.{{if $eleqne}}El{{end}}{{.Name}}Scalar(t, other, leftTensor, opts...); err != nil {
			err = errors.Wrapf(err, "Unable to do {{.Name}}Scalar()")
			return
		}
		if retVal, ok = ret.(*Dense); !ok {
			err = errors.Errorf("Unable to do {{.Name}} - Expected a %T. Got %T instead", retVal, ret)
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
