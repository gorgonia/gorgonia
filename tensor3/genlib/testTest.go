package main

import (
	"fmt"
	"io"
	"text/template"
)

const testHeaderRaw = `func newGenerator(of Dtype) quick.Generator {
	switch of.Kind() {
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		return QCDense{{short .}}(nil)
		{{end -}}
	{{end -}}
	}
	panic("Unreacheable")
}

type Densor interface{
	D() *Dense
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
func (q *QCDense{{short .}}) D() *Dense{return q.Dense}
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
	testHeader *template.Template
	testQC     *template.Template
)

func init() {
	testQC = template.Must(template.New("testQCs").Funcs(funcs).Parse(testQCRaw))
	testHeader = template.Must(template.New("TestHeader").Funcs(funcs).Parse(testHeaderRaw))
}

func testtest(f io.Writer, generic *ManyKinds) {
	// testHeader.Execute(f, generic)
	fmt.Fprint(f, "\n")
	for _, k := range generic.Kinds {
		if isNumber(k) {
			testQC.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
