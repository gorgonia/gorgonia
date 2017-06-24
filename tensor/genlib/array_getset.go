package main

import (
	"fmt"
	"io"
	"text/template"
)

const asSliceRaw = `func (h *header) {{asType . | strip}}s() []{{asType .}} { return *(*[]{{asType .}})(unsafe.Pointer(h)) }
`

const setBasicRaw = `func (h *header) set{{short . }}(i int, x {{asType . }}) { h.{{sliceOf .}}[i] = x }
`

const getBasicRaw = `func (h *header) get{{short .}}(i int) {{asType .}} { return h.{{lower .String | clean | strip }}s()[i]}
`

const getRaw = `// Get returns the ith element of the underlying array of the *Dense tensor.
func (a *array) Get(i int) interface{} {
	switch a.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		return a.get{{short .}}(i)
		{{end -}}
	{{end -}}
	default:
		at := uintptr(a.ptr) + uintptr(i) * a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(at))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

`
const setRaw = `// Set sets the value of the underlying array at the index i. 
func (a *array) Set(i int, x interface{}) {
	switch a.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		xv := x.({{asType .}})
		a.set{{short .}}(i, xv)
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.ptr)
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

`

var (
	AsSlice   *template.Template
	SimpleSet *template.Template
	SimpleGet *template.Template
	Get       *template.Template
	Set       *template.Template
)

func init() {
	AsSlice = template.Must(template.New("AsSlice").Funcs(funcs).Parse(asSliceRaw))
	SimpleSet = template.Must(template.New("SimpleSet").Funcs(funcs).Parse(setBasicRaw))
	SimpleGet = template.Must(template.New("SimpleGet").Funcs(funcs).Parse(getBasicRaw))
	Get = template.Must(template.New("Get").Funcs(funcs).Parse(getRaw))
	Set = template.Must(template.New("Set").Funcs(funcs).Parse(setRaw))
}

func arrayGetSet(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}

	Set.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Get.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
}
