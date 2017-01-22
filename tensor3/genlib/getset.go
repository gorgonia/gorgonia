package main

import (
	"reflect"
	"text/template"
)

const asSliceRaw = `func (t *Dense) {{lower .String | clean | strip}}s() []{{lower .String | clean}} { return *(*[]{{lower .String | clean}})(unsafe.Pointer(t.hdr)) }
`
const setBasicRaw = `func (t *Dense) set{{short .}}(i int, x {{lower .String | clean }}) { t.{{lower .String | clean | strip}}s()[i] = x }
`
const getBasicRaw = `func (t *Dense) get{{short .}}(i int) {{lower .String | clean }} { return t.{{lower .String | clean | strip }}s()[i]}
`
const getRaw = `func (t *Dense) get(i int) interface{} {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{.String | title |  strip}}:
		return t.{{lower .String | clean | strip }}s()[i]
		{{end -}}
	{{end -}}
	default:
		at := uintptr(t.data) + uintptr(i) * t.t.Size()
		val := reflect.NewAt(t.t, unsafe.Pointer(at))
		val = reflect.Indirect(val)
		return val.Interface()
	}
}

`
const setRaw = `func (t *Dense) set(i int, x interface{}) {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{.String | title | strip}}:
		xv := x.({{lower .String | clean}})
		t.{{lower .String | clean|strip}}s()[i] = xv
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.data)
		want := ptr + uintptr(i)*t.t.Size()
		val := reflect.NewAt(t.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

`

const makeDataRaw = `func (t *Dense) makeArray(size int) {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case reflect.{{.String | title | strip}}:
		arr := make([]{{.String | lower | clean }}, size)
		t.fromSlice(arr)
		{{end -}}
	{{end -}}
	default:

	}
}

`

const copyRaw = `func copyDense(dest, src *Dense) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}
	switch dest.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case reflect.{{.String | title | strip}}:
		return copy(dest.{{lower .String | clean | strip}}s(), src.{{lower .String | clean | strip}}s())
		{{end -}}
	{{end -}}
	default:
		dv := reflect.ValueOf(dest.v)
		sv := reflect.ValueOf(src.v)
		return reflect.Copy(dv, sv)
	}
}
`

const copyIterRaw = `func copyDenseIter(dest, src *Dense) int {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	siter := NewFlatIterator(src.AP)
	diter := NewFlatIterator(dest.AP)
	
	k := dest.t.Kind()
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next() ; err != nil {
			if _, ok := err.(NoOpError); !ok {
				panic(err)
			}
			err = nil
			break
		}
		if j, err = siter.Next() ; err != nil {
			if _, ok := err.(NoOpError); !ok {
				panic(err)
			}
			err = nil
			break
		}
		switch k {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case reflect.{{.String | title| strip}}:
			dest.set{{short .}}(i, src.get{{short .}}(j))
			{{end -}}
		{{end -}}
		default:
			dest.set(i, src.get(j))
		}
		count++
	}
	return count
}
`

type ManyKinds struct {
	Kinds []reflect.Kind
}

var (
	AsSlice   *template.Template
	SimpleSet *template.Template
	SimpleGet *template.Template
	Get       *template.Template
	Set       *template.Template
	MakeData  *template.Template
	Copy      *template.Template
	CopyIter  *template.Template
)

func init() {
	AsSlice = template.Must(template.New("AsSlice").Funcs(funcs).Parse(asSliceRaw))
	SimpleSet = template.Must(template.New("SimpleSet").Funcs(funcs).Parse(setBasicRaw))
	SimpleGet = template.Must(template.New("SimpleGet").Funcs(funcs).Parse(getBasicRaw))
	Get = template.Must(template.New("Get").Funcs(funcs).Parse(getRaw))
	Set = template.Must(template.New("Set").Funcs(funcs).Parse(setRaw))
	MakeData = template.Must(template.New("makedata").Funcs(funcs).Parse(makeDataRaw))
	Copy = template.Must(template.New("copy").Funcs(funcs).Parse(copyRaw))
	CopyIter = template.Must(template.New("copyIter").Funcs(funcs).Parse(copyIterRaw))

}
