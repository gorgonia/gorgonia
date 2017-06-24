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

var (
	SimpleSet *template.Template
	SimpleGet *template.Template
)

func init() {
	SimpleSet = template.Must(template.New("SimpleSet").Funcs(funcs).Parse(setBasicRaw))
	SimpleGet = template.Must(template.New("SimpleGet").Funcs(funcs).Parse(getBasicRaw))
}

func genericGetSet(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			AsSlice.Execute(f, k)
			SimpleSet.Execute(f, k)
			SimpleGet.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
