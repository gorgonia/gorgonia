package main

import (
	"fmt"
	"io"
	"text/template"
)

const asSliceRaw = `func (h *Header) {{asType . | strip | title}}s() []{{asType .}} { return *(*[]{{asType .}})(unsafe.Pointer(h)) }
`

const setBasicRaw = `func (h *Header) Set{{short . }}(i int, x {{asType . }}) { h.{{sliceOf .}}[i] = x }
`

const getBasicRaw = `func (h *Header) Get{{short .}}(i int) {{asType .}} { return h.{{lower .String | clean | strip | title }}s()[i]}
`

const getRaw = `// Get returns the ith element of the underlying array of the *Dense tensor.
func (a *array) Get(i int) interface{} {
	switch a.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		return a.{{getOne .}}(i)
		{{end -}}
	{{end -}}
	default:
		at := uintptr(a.Ptr) + uintptr(i) * a.t.Size()
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
		a.{{setOne .}}(i, xv)
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(a.Ptr)
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
}

`

const memsetRaw = `// Memset sets all values in the array.
func (a *array) Memset(x interface{}) error {
	switch a.t {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		if xv, ok := x.({{asType .}}); ok {
			data := a.{{sliceOf .}}
			for i := range data{
				data[i] = xv
			}
			return nil
		}

		{{end -}}
	{{end -}}
	}
	
	xv := reflect.ValueOf(x)
	ptr := uintptr(a.Ptr)
	for i := 0; i < a.L; i++ {
		want := ptr + uintptr(i)*a.t.Size()
		val := reflect.NewAt(a.t, unsafe.Pointer(want))
		val = reflect.Indirect(val)
		val.Set(xv)
	}
	return nil
}
`

const arrayEqRaw = ` // Eq checks that any two arrays are equal
func (a array) Eq(other interface{}) bool {
	if oa, ok := other.(*array); ok {
		if oa.t != a.t {
			return false
		}

		if oa.L != a.L {
			return false
		}

		if oa.C != a.C {
			return false
		}

		// same exact thing
		if uintptr(oa.Ptr) == uintptr(a.Ptr){
			return true
		}

		switch a.t.Kind() {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case reflect.{{reflectKind .}}:
			for i, v := range a.{{sliceOf .}} {
				if oa.{{getOne .}}(i) != v {
					return false
				}
			}
			{{end -}}
		{{end -}}
		default:
			for i := 0; i < a.L; i++{
				if !reflect.DeepEqual(a.Get(i), oa.Get(i)){
					return false
				}
			}
		}
		return true
	}
	return false
}`

const copyArrayIterRaw = `func copyArrayIter(dst, src array, diter, siter Iterator) (count int, err error){
	if dst.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil {
		return copyArray(dst, src), nil
	}

	if (diter != nil && siter == nil) || (diter == nil && siter != nil) {
		return 0, errors.Errorf("Cannot copy array when only one iterator was passed in")
	}

	k := dest.t.Kind()
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = diter.NextValidity(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		if j, validj, err = siter.NextValidity(); err != nil {
			if err = handleNoOp(err); err != nil {
				return count, err
			}
			break
		}
		switch k {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case reflect.{{reflectKind .}}:
			dest.{{setOne .}}(i, src.{{getOne .}}(j))
			{{end -}}
		{{end -}}
		default:
			dest.Set(i, src.Get(j))
		}
		count++
	}

}
`

const memsetIterRaw = `
func (t *array) memsetIter(x interface{}, it Iterator) (err error) {
	var i int
	switch t.t{
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		xv, ok := x.({{asType .}})
		if !ok {
			return errors.Errorf(dtypeMismatch, t.t, x)
		}
		data := t.{{sliceOf .}}
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			data[i] = xv	
		}
		err = handleNoOp(err)
		{{end -}}
	{{end -}}
	default:
		xv := reflect.ValueOf(x)
		ptr := uintptr(t.Ptr)
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(xv)
		}
		err = handleNoOp(err)
	}
	return
}

`

const zeroIterRaw = `func (t *array) zeroIter(it Iterator) (err error){
	var i int
	switch t.t {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case {{reflectKind .}}:
		data := t.{{sliceOf .}}
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			data[i] = {{if eq .String "bool" -}}
				false
			{{else if eq .String "string" -}}""
			{{else if eq .String "unsafe.Pointer" -}}nil
			{{else -}}0{{end -}}
		}
		err = handleNoOp(err)
		{{end -}}
	{{end -}}
	default:
		ptr := uintptr(t.Ptr)
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			want := ptr + uintptr(i)*t.t.Size()
			val := reflect.NewAt(t.t, unsafe.Pointer(want))
			val = reflect.Indirect(val)
			val.Set(reflect.Zero(t.t))
		}
		err = handleNoOp(err)
	}
	return
}
`

var (
	AsSlice    *template.Template
	SimpleSet  *template.Template
	SimpleGet  *template.Template
	Get        *template.Template
	Set        *template.Template
	Memset     *template.Template
	MemsetIter *template.Template
	Eq         *template.Template
	ZeroIter   *template.Template
)

func init() {
	AsSlice = template.Must(template.New("AsSlice").Funcs(funcs).Parse(asSliceRaw))
	SimpleSet = template.Must(template.New("SimpleSet").Funcs(funcs).Parse(setBasicRaw))
	SimpleGet = template.Must(template.New("SimpleGet").Funcs(funcs).Parse(getBasicRaw))
	Get = template.Must(template.New("Get").Funcs(funcs).Parse(getRaw))
	Set = template.Must(template.New("Set").Funcs(funcs).Parse(setRaw))
	Memset = template.Must(template.New("Memset").Funcs(funcs).Parse(memsetRaw))
	MemsetIter = template.Must(template.New("MemsetIter").Funcs(funcs).Parse(memsetIterRaw))
	Eq = template.Must(template.New("ArrayEq").Funcs(funcs).Parse(arrayEqRaw))
	ZeroIter = template.Must(template.New("Zero").Funcs(funcs).Parse(zeroIterRaw))
}

func generateArrayMethods(f io.Writer, ak Kinds) {
	Set.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Get.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Memset.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	MemsetIter.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	Eq.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
	ZeroIter.Execute(f, ak)
	fmt.Fprintf(f, "\n\n\n")
}

func generateHeaderGetSet(f io.Writer, ak Kinds) {
	for _, k := range ak.Kinds {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
