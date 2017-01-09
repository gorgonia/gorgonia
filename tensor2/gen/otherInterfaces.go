package main

import "text/template"

const sliceRaw = `func (a {{.Name}}) Slice(s Slice) (Array, error){
	start, end, _, err := SliceDetails(s, len(a))
	if err != nil {
		return nil, err
	}
	return a[start:end], nil
}
`

const dtyperRaw = `func (a {{.Name}}) Dtype() Dtype {	return {{title .Of}}}
`

var (
	sliceTmpl *template.Template
	dtypeTmpl *template.Template
)

func init() {
	sliceTmpl = template.Must(template.New("Slice").Funcs(funcMap).Parse(sliceRaw))
	dtypeTmpl = template.Must(template.New("Dtyper").Funcs(funcMap).Parse(dtyperRaw))
}
