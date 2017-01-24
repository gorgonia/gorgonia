package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

var cmpBinOps = []struct {
	OpName string
	OpSymb string
	is     func(reflect.Kind) bool
}{
	{"Eq", "==", isEq},
	{"Gt", ">", isOrd},
	{"Gte", ">=", isOrd},
	{"Lt", "<", isOrd},
	{"Lte", "<=", isOrd},
}

const eleqordDDRaw = `func (t *Dense) {{lower .OpName}}DD(other *Dense, same bool) (retVal *Dense, err error) {
	k := t.t.Kind()
	if k != other.t.Kind() {
		err = errors.Errorf(typeMismatch, t.t, other.t)
		return
	}
	if t.len() != other.len() {
		err = errors.Errorf(lenMismatch, t.len(), other.len())
	}
	{{$op := .OpName}}
	retVal := recycledDenseNoFix(t.t, t.Shape().Clone())
	switch k {
	{{range .Kinds -}}
		{{ $eq := isEq . -}}
		{{ $ord := isOrd . -}}
		{{if or $eq $ord -}} 
	case reflect.{{reflectKind .}}:
		td := t.{{asType . | strip}}s()
		od := other.{{asType . | strip}}s()
		{{if isNumber . -}}
			if same {
				ret := {{lower $op}}DDSame{{short .}}(td, od)
				retVal.fromSlice(ret)
			} else {
				ret := {{lower $op}}DDBools{{short .}}(td, od)
				retVal.fromSlice(ret)
			}
		{{else -}}
			ret := {{lower $op}}DDBools{{short .}}(td, od)
			retVal.fromSlice(ret)
		{{end -}}

		{{end -}}
	{{end -}}
	default:
		err = errors.Errorf(unsupportedDtype, d.t, "{{lower .OpName}}")
		return
	}
	retVal.fix()
	err = retVal.sanity()
	return
}

`

var (
	ddElEqOrd *template.Template
)

func init() {
	ddElEqOrd = template.Must(template.New("ElEqOrdDD").Funcs(funcs).Parse(eleqordDDRaw))
}

func denseCmp(f io.Writer, generic *ManyKinds) {
	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := ArithBinOps{generic, bo.OpName, bo.OpSymb, false}
		ddElEqOrd.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}
}
