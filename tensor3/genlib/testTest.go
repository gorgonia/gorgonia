package main

import (
	"fmt"
	"io"
	"text/template"
)

const testHeader = `func newGenerator(t *Dense) quick.Generator {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		return QCDense{{short .}}{t}
		{{end -}}
	{{end -}}
	}
	panic("Unreacheable")
}

var numbers = []Dtype{
	{{range .Kinds -}}
		{{if isNumber . -}}{{asType . | title}}, {{end -}}
	{{end -}}
}
`

const testQCRaw = `type QCDense{{short .}} struct {
	*Dense 
}

func (*QCDense{{short .}}) Generate(r *rand.Rand, size int) reflect.Value {
	s := make([]{{asType .}}, size)
	for i := range s {
		{{if hasPrefix .String "uint" -}}
			s[i] = {{asType .}}(r.Uint32())
		{{else if hasPrefix .String "int" -}}
			s[i] = {{asType .}}(r.Int())
		{{else if eq .String "float64" -}}
			s[i] = r.Float64()
		{{else if eq .String "float32" -}}
			s[i] = r.Float32()
		{{else if eq .String "complex64" -}}
			s[i] = complex(r.Float32(), r.Float32())
		{{else if eq .String "complex128" -}}
			s[i] = complex(r.Float64(), r.Float64())
		{{end -}}
	}
	d := recycledDense({{asType . | title | strip}}, Shape{size}, WithBacking(s))
	q := new(QCDense{{short .}})
	q.Dense = d
	return reflect.ValueOf(q)
}
`

var (
	testQC *template.Template
)

func init() {
	testQC = template.Must(template.New("testQCs").Funcs(funcs).Parse(testQCRaw))
}

func testtest(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isNumber(k) {
			testQC.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
