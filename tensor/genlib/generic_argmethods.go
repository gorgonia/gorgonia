package main

import (
	"fmt"
	"io"
	"text/template"
)

const argmaxRaw = `func argmax{{short .}}(a []{{asType .}}) int {
	var f {{asType .}}
	var max int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			max = i
			set = true

			continue
		}

		{{if hasPrefix .String "float" -}}
			if {{mathPkg .}}IsNaN(v) || {{mathPkg .}}IsInf(v, 1) {
				max = i
				return max
			}
		{{end -}}
		if v > f {
			max = i
			f = v
		}
	}
	return max
}
`

const argminRaw = `func argmin{{short .}}(a []{{asType .}}) int {
	var f {{asType .}}
	var min int
	var set bool

	for i, v := range a {
		if !set {
			f = v
			min = i
			set = true

			continue
		}
		{{if hasPrefix .String "float" -}}
			if {{mathPkg .}}IsNaN(v) || {{mathPkg .}}IsInf(v, -1) {
				min = i
				return min
			}
		{{end -}}
		if v < f {
			min = i
			f = v
		}
	}
	return min
}

`

var (
	argmin *template.Template
	argmax *template.Template
)

func init() {
	argmin = template.Must(template.New("argmin").Funcs(funcs).Parse(argminRaw))
	argmax = template.Must(template.New("argmax").Funcs(funcs).Parse(argmaxRaw))
}

func genericArgmethods(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isNumber(k) && isOrd(k) {
			fmt.Fprintf(f, "/* %s */\n\n", k)
			argmin.Execute(f, k)
			argmax.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
