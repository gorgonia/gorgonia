package main

import (
	"io"
	"text/template"
)

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
	switch dest.t {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case {{reflectKind .}}:
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

	dt := dest.t
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
		
		switch dt {
		{{range .Kinds -}}
			{{if isParameterized . -}}
			{{else -}}
		case {{reflectKind .}}:
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
	switch t.t {
	{{range .Kinds -}}
		{{if isParameterized .}}
		{{else -}}
	case {{reflectKind .}}:
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
	CopySliced *template.Template
	CopyIter   *template.Template
	Slice      *template.Template
)

func init() {

	CopySliced = template.Must(template.New("copySliced").Funcs(funcs).Parse(copySlicedRaw))
	CopyIter = template.Must(template.New("copyIter").Funcs(funcs).Parse(copyIterRaw))
	Slice = template.Must(template.New("slice").Funcs(funcs).Parse(sliceRaw))
}

func generateDenseGetSet(f io.Writer, generic Kinds) {

	// CopySliced.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
	// CopyIter.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
	// Slice.Execute(f, generic)
	// fmt.Fprintf(f, "\n\n\n")
}
