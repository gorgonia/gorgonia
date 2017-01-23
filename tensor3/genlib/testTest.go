package main

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

func (t QCDense{{short .}}) Generate(rand *rand.Rand, size int) reflect.Value {
	ret := new(Dense)
	data := make([]{{asType .}}, size)
	ret.fromSlice(data)
	return reflect.ValueOf(ret)
}
`
