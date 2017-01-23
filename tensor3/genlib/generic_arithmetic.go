package main

import (
	"fmt"
	"io"
	"text/template"
)

const genericVecVecArithRaw = `func {{lower .OpName}}{{short .Kind}}(a, b []{{asType .Kind}}) error {
	if len(a) != len(b){
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	{{if hasPrefix .Kind.String "float" -}}
		vec{{short .Kind | lower }}.{{.OpName}}(a, b)
	{{else if hasPrefix .Kind.String "complex" -}} 
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices{{end}}
		for i, v := range b {
			{{if or $scaleInv $div -}}
			if v == {{asType .Kind}}(0) {
				errs = append(errs, i)
				a[i] = 0
				continue
			}

			{{end -}}
			{{if .IsFunc -}}
				a[i] = {{asType .Kind}}({{.OpSymb}} (float64(a[i]), float64(v)))
			{{else -}}
				a[i] {{.OpSymb}}= v 
			{{end -}}
		}

		{{if or $scaleInv $div -}}
			if errs != nil {
				return errs
			}
		{{end -}}
	{{end -}}
	return nil
}
`
const genericVecScalarArithRaw = `func {{lower .OpName}}{{short .Kind}}(a []{{asType .Kind}}, b {{asType .Kind}}) error {
	{{if hasPrefix .Kind.String "float" -}}
		vec{{short .Kind | lower }}.{{.OpName}}([]{{asType .Kind}}(a), b)
	{{else if hasPrefix .Kind.String "complex" -}} 
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices{{end}}
		for i, v := range a {
			{{if or $scaleInv $div -}}
			if v == {{asType .Kind}}(0) {
				errs = append(errs, i)
				a[i] = 0 
				continue
			}
			{{end -}}

			a[i] = {{if hasSuffix .OpName "R" -}}
				{{if .IsFunc -}} 
					{{asType .Kind}}({{.OpSymb}}(float64(b), float64(v)))
				{{else -}} 
					b {{.OpSymb}} v 
				{{end -}}
			{{else -}} 
				{{if .IsFunc -}} 
					{{asType .Kind}}({{.OpSymb}}(float64(v), float64(b)))
				{{else -}} 
					v {{.OpSymb}} b
				{{end -}}
			{{end -}}
		}
		{{if or $scaleInv $div -}}
			if errs != nil {
				return errs
			}
		{{end -}}
	{{end -}}
	return nil
}
`

var (
	genericVecScalarArith *template.Template
	genericVecVecArith    *template.Template
)

func init() {
	genericVecVecArith = template.Must(template.New("vecvecArith").Funcs(funcs).Parse(genericVecVecArithRaw))
	genericVecScalarArith = template.Must(template.New("vecscalarArith").Funcs(funcs).Parse(genericVecScalarArithRaw))
}

func genericArith(f io.Writer, generic *ManyKinds) {
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				genericVecVecArith.Execute(f, op)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}

	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				genericVecScalarArith.Execute(f, op)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}
}
