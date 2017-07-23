package main

import (
	"fmt"
	"io"
	"text/template"
)

const memsetIterRaw = `
func (t *Dense) memsetIter(x interface{}) (err error) {
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
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
		ptr := uintptr(t.array.Ptr)
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

const zeroRaw = `func (t *Dense) zeroIter() (err error){
	it := NewFlatIterator(t.AP)
	var i int
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized . -}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		data := t.{{sliceOf .}}
		for i, err = it.Next(); err == nil; i, err = it.Next(){
			data[i] = {{if eq .String "bool" -}}
				false
			{{else if eq .String "string" -}}""
			{{else if eq .String "unsafe.Pointer" -}}nil
			{{else -}}0{{end}}
		}
		err = handleNoOp(err)
		{{end -}}
	{{end -}}
	default:
		ptr := uintptr(t.array.Ptr)
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

const copySlicedRaw = `func copySliced(dest *Dense, dstart, dend int, src *Dense, sstart, send int) int{
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if src.IsMasked(){
		mask:=dest.mask
		if cap(dest.mask) < dend{
			mask = make([]bool, dend)
		}
		copy(mask, dest.mask)
		dest.mask=mask
		copy(dest.mask[dstart:dend], src.mask[sstart:send])
	}
	switch dest.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		return copy(dest.{{sliceOf .}}[dstart:dend], src.{{sliceOf .}}[sstart:send])
		{{end -}}
	{{end -}}
	default:
		dv := reflect.ValueOf(dest.v)
		dv = dv.Slice(dstart, dend)
		sv := reflect.ValueOf(src.v)
		sv = sv.Slice(sstart, send)
		return reflect.Copy(dv, sv)
	}	
}
`

const copyIterRaw = `func copyDenseIter(dest, src *Dense, diter, siter *FlatIterator) (int, error) {
	if dest.t != src.t {
		panic("Cannot copy arrays of different types")
	}

	if diter == nil && siter == nil && !dest.IsMaterializable() && !src.IsMaterializable() {
		return copyDense(dest, src), nil
	}

	if diter == nil {
		diter = NewFlatIterator(dest.AP)	
	}
	if siter == nil {
		siter = NewFlatIterator(src.AP)
	}
	
	isMasked:= src.IsMasked()
	if isMasked{
		if cap(dest.mask)<src.DataSize(){
			dest.mask=make([]bool, src.DataSize())
		}
		dest.mask=dest.mask[:dest.DataSize()]
	}

	k := dest.t.Kind()
	var i, j, count int
	var err error
	for {
		if i, err = diter.Next() ; err != nil {
			if err = handleNoOp(err); err != nil{
				return count, err
			}
			break
		}
		if j, err = siter.Next() ; err != nil {
			if err = handleNoOp(err); err != nil{
				return count, err
			}
			break
		}
		if isMasked{
			dest.mask[i]=src.mask[j]
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
	return count, err
}
`

const sliceRaw = `// the method assumes the AP and metadata has already been set and this is simply slicing the values
func (t *Dense) slice(start, end int) {
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case reflect.{{reflectKind .}}:
		data := t.{{sliceOf .}}[start:end]
		t.fromSlice(data)
		{{end -}}
	{{end -}}
	default:
		v := reflect.ValueOf(t.v)
		v = v.Slice(start, end)
		t.fromSlice(v.Interface())
	}	
}
`

var (
	MemsetIter *template.Template
	Zero       *template.Template
	CopySliced *template.Template
	CopyIter   *template.Template
	Slice      *template.Template
)

func init() {
	MemsetIter = template.Must(template.New("MemsetIter").Funcs(funcs).Parse(memsetIterRaw))
	Zero = template.Must(template.New("Zero").Funcs(funcs).Parse(zeroRaw))
	CopySliced = template.Must(template.New("copySliced").Funcs(funcs).Parse(copySlicedRaw))
	CopyIter = template.Must(template.New("copyIter").Funcs(funcs).Parse(copyIterRaw))
	Slice = template.Must(template.New("slice").Funcs(funcs).Parse(sliceRaw))
}

func generateDenseGetSet(f io.Writer, generic Kinds) {
	MemsetIter.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Zero.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	CopySliced.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	CopyIter.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
	Slice.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
}
