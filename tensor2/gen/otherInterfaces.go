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

// this is just generated for the dummy types
const floatArrRaw = `func (a {{.Name}}) HasNaN() bool{ return false}
func (a {{.Name}}) HasInf() bool{ return false}
`

var (
	sliceTmpl    *template.Template
	dtypeTmpl    *template.Template
	floatArrTmpl *template.Template
)

func init() {
	sliceTmpl = template.Must(template.New("Slice").Funcs(funcMap).Parse(sliceRaw))
	dtypeTmpl = template.Must(template.New("Dtyper").Funcs(funcMap).Parse(dtyperRaw))
	floatArrTmpl = template.Must(template.New("Float").Funcs(funcMap).Parse(floatArrRaw))
}
