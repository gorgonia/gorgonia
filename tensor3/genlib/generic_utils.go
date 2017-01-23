package main

import (
	"io"
	"text/template"
)

const rangeRaw = `// Range creates a ranged array with a given type. It panics if the Dtype is not supported or does not represent a naturally orderable type (strings, pointers etc)
// Do note that the range algorithm is very simple, and simply does increments or decrements of 1. This means for floating point types
// you're not able to create a range with a 0.1 increment step, and for complex number types, the imaginary part will always be 0i
func Range(dt Dtype, start, end int) interface{} {
	size := end - start
	incr := true
	if start > end {
		incr = false
		size = start - end
	}

	if size < 0 {
		panic("Cannot create a range that is negative in size")
	}
	switch dt.Kind(){
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
		{{if isRangeable . -}}
	case reflect.{{reflectKind .}}:
		{{if hasPrefix .String "float" -}}
			return vec{{short . | lower}}.Range(start, end)
		{{else -}}
			retVal := make([]{{asType .}}, size)
			{{if eq .String "complex64" -}}
			for i, v := 0, complex(float32(start), float32(0.0)); i < size; i++ {
			{{else if eq .String "complex128" -}}
			for i, v := 0, complex(float64(start), float64(0.0)); i < size; i++ {
			{{else -}}
			for i, v := 0, {{asType .}}(start); i < size; i++ {
			{{end -}}
				retVal[i] = v
				if incr {
					v++
				} else{
					v--
				}
			}
			return retVal
		{{end -}}
		{{end -}}
		{{end -}}
	{{end -}}
	default:
		err := errors.Errorf("Unrangeable Type %v", dt)
		panic(err)
	}
}
`

var (
	Range *template.Template
)

func init() {
	Range = template.Must(template.New("Range").Funcs(funcs).Parse(rangeRaw))
}

func utils(f io.Writer, generic *ManyKinds) {
	Range.Execute(f, generic)
}
