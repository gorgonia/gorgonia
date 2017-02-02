package main

import (
	"fmt"
	"io"
	"text/template"
)

const genericVecVecArithRaw = `func vec{{.OpName}}{{short .Kind}}(a, b []{{asType .Kind}}) error {
	if len(a) != len(b){
		return errors.Errorf(lenMismatch, len(a), len(b))
	}

	{{if hasPrefix .Kind.String "float" -}}
		vec{{short .Kind | lower }}.{{.OpName}}(a, b)
	{{else if hasPrefix .Kind.String "complex" -}}
		for i, v := range b {
			{{if eq .OpName "Pow" -}}
				a[i] = {{asType .Kind}}(cmplx.Pow(complex128(a[i]), complex128(v)))
			{{else -}}
				a[i] {{.OpSymb}}= v 
			{{end -}}
			}
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices
		{{end -}}
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
const genericVecScalarArithRaw = `func {{if .IsIncr}}incr{{.OpName}}{{else}}{{lower .OpName}}{{end}}{{short .Kind}}(a{{if .IsIncr}}, incr{{end}} []{{asType .Kind}}, b {{asType .Kind}}) error {
	{{if hasPrefix .Kind.String "float" -}}
		{{if .IsIncr -}}
		vec{{short .Kind | lower}}.Incr{{.OpName}}(a, b, incr)
		{{else -}}
			vec{{short .Kind | lower }}.{{.OpName}}(a, b)
		{{end -}}
	{{else if hasPrefix .Kind.String "complex" -}} 
		for i, v := range a {
			{{if hasPrefix .OpName "Pow" -}}
				{{if .IsIncr}}incr[i] +{{else}}a[i] {{end}}=  {{if hasSuffix .OpName "R" -}}   {{asType .Kind}}(cmplx.Pow(complex128(b), complex128(v)))  {{else -}}  {{asType .Kind}}(cmplx.Pow(complex128(v), complex128(b))) {{end -}}
			{{else -}}
			 	{{if .IsIncr}}incr[i] +{{else}}a[i] {{end}}= {{if hasSuffix .OpName "R" -}} b {{.OpSymb}} v {{else -}} v {{.OpSymb}} b {{end -}}
			{{end -}}
			}
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices
		{{end -}}
		for i, v := range a {
			{{if or $scaleInv $div -}}
			if v == {{asType .Kind}}(0) {
				errs = append(errs, i)
				a[i] = 0 
				continue
			}

			{{end -}}
			{{if .IsIncr}}incr[i] +{{else}}a[i] {{end}}= {{if hasSuffix .OpName "R" -}}
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

// scalar scalar is for reduction
const genericScalarScalarArithRaw = `func {{lower .OpName}}{{short .Kind}}(a, b {{asType .Kind}}) (c {{asType .Kind}}) {return a {{.OpSymb}} b}
`

var (
	genericVecScalarArith    *template.Template
	genericVecVecArith       *template.Template
	genericScalarScalarArith *template.Template
)

func init() {
	genericVecVecArith = template.Must(template.New("vecvecArith").Funcs(funcs).Parse(genericVecVecArithRaw))
	genericVecScalarArith = template.Must(template.New("vecscalarArith").Funcs(funcs).Parse(genericVecScalarArithRaw))
	genericScalarScalarArith = template.Must(template.New("scalscalArith").Funcs(funcs).Parse(genericScalarScalarArithRaw))
}

type IncrOp struct {
	ArithBinOp
	IsIncr bool
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
				incrOp := IncrOp{op, false}
				genericVecScalarArith.Execute(f, incrOp)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}

	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* incr %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrOp := IncrOp{op, true}
				genericVecScalarArith.Execute(f, incrOp)
				fmt.Fprintln(f, "\n")
			}
		}
		fmt.Fprintln(f, "\n")
	}

	// generic scalar-scalar
	for _, k := range generic.Kinds {
		if isNumber(k) {
			op := ArithBinOp{k, "Add", "+", false}
			genericScalarScalarArith.Execute(f, op)
		}
	}
}
