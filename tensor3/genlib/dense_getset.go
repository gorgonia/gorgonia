package main

import (
	"fmt"
	"io"
	"text/template"
)

const asSliceRaw = `func (t *Dense) {{asType . | strip }}s() []{{asType .}} { return *(*[]{{asType .}})(unsafe.Pointer(t.hdr)) }
`
const setBasicRaw = `func (t *Dense) set{{short .}}(i int, x {{asType .}}) { t.{{asType . | strip}}s()[i] = x }
`
const getBasicRaw = `func (t *Dense) get{{short .}}(i int) {{asType .}} { return t.{{lower .String | clean | strip }}s()[i]}
`
const getRaw = `func (t *Dense) get(i int) interface{} {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		return t.get{{short .}}(i)
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
	case reflect.{{reflectKind .}}:
		xv := x.({{asType .}})
		t.set{{short .}}(i, xv)
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

const memsetRaw = `func (t *Dense) Memset(x interface{}) error {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		xv, ok := x.({{asType .}})
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.{{asType . | strip}}s()
		for i := range data{
			data[i] = xv
		}
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.data)
		for i := 0; i < t.hdr.Len; i++ {
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
	}
	return nil
}

`

const makeDataRaw = `func (t *Dense) makeArray(size int) {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case reflect.{{reflectKind .}}:
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
	case reflect.{{reflectKind .}}:
		return copy(dest.{{asType . | strip}}s(), src.{{asType . | strip}}s())
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
		case reflect.{{reflectKind .}}:
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

var (
	AsSlice   *template.Template
	SimpleSet *template.Template
	SimpleGet *template.Template
	Get       *template.Template
	Set       *template.Template
	Memset    *template.Template
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
	Memset = template.Must(template.New("Memset").Funcs(funcs).Parse(memsetRaw))
	MakeData = template.Must(template.New("makedata").Funcs(funcs).Parse(makeDataRaw))
	Copy = template.Must(template.New("copy").Funcs(funcs).Parse(copyRaw))
	CopyIter = template.Must(template.New("copyIter").Funcs(funcs).Parse(copyIterRaw))

}

func getset(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
	MakeData.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Set.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Get.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Memset.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Copy.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	CopyIter.Execute(f, generic)
}
