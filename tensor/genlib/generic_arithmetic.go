package main

import (
	"fmt"
	"io"
	"text/template"
)

const genericVecVecArithRaw = `func {{if .IsIncr}}incrVec{{else}}vec{{end}}{{.OpName}}{{short .Kind}}(a, b{{if .IsIncr}}, incr{{end}} []{{asType .Kind}}) error {
	if len(a) != len(b){
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	a = a[:len(a)]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks
	
	{{if .IsIncr -}}
	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	{{end -}}
	{{if hasPrefix .Kind.String "float" -}}
		{{if .IsIncr -}}
			vec{{short .Kind | lower}}.Incr{{.OpName}}(a, b, incr)
		{{else -}}
			vec{{short .Kind | lower }}.{{.OpName}}(a, b)
		{{end -}}
	{{else if hasPrefix .Kind.String "complex" -}}
		for i, v := range b {
			{{if .IsIncr -}}
				{{if eq .OpName "Pow" -}}
					incr[i] += {{if eq .Kind.String "complex64"}}{{asType .Kind}}(cmplx.Pow(complex128(a[i]), complex128(v))){{else}}cmplx.Pow(a[i], v){{end}}
				{{else -}}
					incr[i] += a[i] {{.OpSymb}} v
				{{end -}}
			{{else -}}
				{{if eq .OpName "Pow" -}}
					a[i] = {{if eq .Kind.String "complex64"}}{{asType .Kind}}(cmplx.Pow(complex128(a[i]), complex128(v))){{else}}cmplx.Pow(a[i], v){{end}}
				{{else -}}
					a[i] {{.OpSymb}}= v 
				{{end -}}
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
				{{if .IsIncr -}}
					incr[i] = 0
				{{else -}}
					a[i] = 0
				{{end -}}
				continue
			}
			{{end -}}
			{{if .IsIncr -}}
				{{if .IsFunc -}}
					incr[i] += {{asType .Kind}}({{.OpSymb}} (float64(a[i]), float64(v)))
				{{else -}}
					incr[i] += a[i] {{.OpSymb}} v 
				{{end -}}
			{{else -}}
				{{if .IsFunc -}}
					a[i] = {{asType .Kind}}({{.OpSymb}} (float64(a[i]), float64(v)))
				{{else -}}
					a[i] {{.OpSymb}}= v 
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

func {{if .IsIncr}}incrVec{{else}}vec{{end}}{{.OpName}}{{short .Kind}}Masked(a, b{{if .IsIncr}}, incr{{end}} []{{asType .Kind}}, mask []bool) error {
	if len(a) != len(b){
		return errors.Errorf(lenMismatch, len(a), len(b))
	}
	if len(mask) != len(a){
		 return errors.Errorf(lenMismatch, len(a), len(b), len(mask))		
		}

	a = a[:len(a)]
	b = b[:len(a)] // these two lines are to eliminate any in-loop bounds checks
	
	{{if .IsIncr -}}
	if len(incr) != len(a) {
		return errors.Errorf(lenMismatch, len(a), len(incr))
	}
	incr = incr[:len(a)]
	{{end -}}

	{{if hasPrefix .Kind.String "complex" -}} 
		for i, v := range b {
				if !mask[i]{
			{{if .IsIncr -}}
				{{if eq .OpName "Pow" -}}
					incr[i] += {{if eq .Kind.String "complex64"}}{{asType .Kind}}(cmplx.Pow(complex128(a[i]), complex128(v))){{else}}cmplx.Pow(a[i], v){{end}}
				{{else -}}
					incr[i] += a[i] {{.OpSymb}} v
				{{end -}}
			{{else -}}
				{{if eq .OpName "Pow" -}}
					a[i] = {{if eq .Kind.String "complex64"}}{{asType .Kind}}(cmplx.Pow(complex128(a[i]), complex128(v))){{else}}cmplx.Pow(a[i], v){{end}}
				{{else -}}
					a[i] {{.OpSymb}}= v 
				{{end -}}
			{{end -}}
			}
		}	
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices
		{{end -}}		
			for i, v := range b {
			if !mask[i]{
			{{if or $scaleInv $div -}}
			if v == {{asType .Kind}}(0) {
				errs = append(errs, i)
				{{if .IsIncr -}}
					incr[i] = 0
				{{else -}}
					a[i] = 0
				{{end -}}
				continue
			}
			{{end -}}

			{{if .IsIncr -}}
				{{if .IsFunc -}}
					incr[i] += {{asType .Kind}}({{.OpSymb}} (float64(a[i]), float64(v)))
				{{else -}}
					incr[i] += a[i] {{.OpSymb}} v 
				{{end -}}
			{{else -}}
				{{if .IsFunc -}}
					a[i] = {{asType .Kind}}({{.OpSymb}} (float64(a[i]), float64(v)))
				{{else -}}
					a[i] {{.OpSymb}}= v 
				{{end -}}
				
			{{end -}}
			}
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

func {{if .IsIncr}}incr{{.OpName}}{{else}}{{lower .OpName}}{{end}}{{short .Kind}}Masked(a{{if .IsIncr}}, incr{{end}} []{{asType .Kind}}, b {{asType .Kind}}, mask []bool) error {
	if len(mask) != len(a){
		 return errors.Errorf(lenMismatch, len(a), len(mask))		
		}
	{{if hasPrefix .Kind.String "complex" -}} 
		for i, v := range a {
			if !mask[i]{
				{{if hasPrefix .OpName "Pow" -}}
					{{if .IsIncr}}incr[i] +{{else}}a[i] {{end}}=  {{if hasSuffix .OpName "R" -}}   {{asType .Kind}}(cmplx.Pow(complex128(b), complex128(v)))  {{else -}}  {{asType .Kind}}(cmplx.Pow(complex128(v), complex128(b))) {{end -}}
				{{else -}}
					{{if .IsIncr}}incr[i] +{{else}}a[i] {{end}}= {{if hasSuffix .OpName "R" -}} b {{.OpSymb}} v {{else -}} v {{.OpSymb}} b {{end -}}
				{{end -}}
				}
			}
	{{else -}}
		{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
		{{$div := hasPrefix .OpName "Div" -}}
		{{if or $scaleInv $div -}}var errs errorIndices
		{{end -}}
		for i, v := range a {
			if !mask[i]{
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
				incrOp := IncrOp{op, false}
				genericVecVecArith.Execute(f, incrOp)
				fmt.Fprint(f, "\n")
			}
		}
		fmt.Fprint(f, "\n")
	}

	for _, bo := range binOps {
		fmt.Fprintf(f, "/* incr %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrOp := IncrOp{op, true}
				genericVecVecArith.Execute(f, incrOp)
				fmt.Fprint(f, "\n")
			}
		}
		fmt.Fprint(f, "\n")
	}

	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrOp := IncrOp{op, false}
				genericVecScalarArith.Execute(f, incrOp)
				fmt.Fprint(f, "\n")
			}
		}
		fmt.Fprint(f, "\n")
	}

	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* incr %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if isNumber(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, bo.IsFunc}
				incrOp := IncrOp{op, true}
				genericVecScalarArith.Execute(f, incrOp)
				fmt.Fprint(f, "\n")
			}
		}
		fmt.Fprint(f, "\n")
	}

	// generic scalar-scalar
	for _, k := range generic.Kinds {
		if isNumber(k) {
			op := ArithBinOp{k, "Add", "+", false}
			genericScalarScalarArith.Execute(f, op)
		}
	}
}
