package main

import "text/template"

// sliceRaw is to generate arrays that implement Slicer
const sliceRaw = `func (a {{.Name}}) Slice(start, end int) (Array, error){
	if end >= len(a) || start < 0 {
		return nil, errors.Errorf(sliceIndexOOB, start, end, len(a))
	}

	return a[start:end], nil
}
`

const dtyperRaw = `func (a {{.Name}}) Dtype() Dtype { return {{title .Of}} }
`

var (
	sliceTmpl *template.Template
	dtypeTmpl *template.Template
)

func init() {
	sliceTmpl = template.Must(template.New("Slice").Funcs(funcMap).Parse(sliceRaw))
	dtypeTmpl = template.Must(template.New("Dtyper").Funcs(funcMap).Parse(dtyperRaw))
}
