package main

import (
	"fmt"
	"io"
	"text/template"
)

const genericVecVecCmpSameRaw = `func {{lower .OpName}}DDSame{{short .Kind}}(a, b []{{asType .Kind}}) (retVal []{{asType .Kind}}) {
	retVal = make([]{{asType .Kind}}, len(a))
	for i, v := range a {
		if v {{.OpSymb}} b[i] {
			retVal[i] = {{if eq .Kind.String "bool"}}true{{else}}1{{end}}
		} else {
			retVal[i] = {{if eq .Kind.String "bool"}}false{{else}}0{{end}}
		}
	}
	return retVal
}
`

const genericVecVecCmpBoolRaw = `func {{lower .OpName}}DDBools{{short .Kind}}(a, b []{{asType .Kind}}) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v {{.OpSymb}} b[i]
	}
	return retVal
}
`

const genericVecScalarCmpSameRaw = `func {{lower .OpName}}DSSame{{short .Kind}}(a []{{asType .Kind}}, b {{asType .Kind}}) (retVal []{{asType .Kind}}) {
	retVal = make([]{{asType .Kind}}, len(a))
	for i, v := range a {
		if v {{.OpSymb}} b {
			retVal[i] = {{if eq .Kind.String "bool"}}true{{else}}1{{end}}
		} else {
			retVal[i] = {{if eq .Kind.String "bool"}}false{{else}}0{{end}}
		}
	}
	return retVal
}
`
const genericVecScalarCmpBoolRaw = `func {{lower .OpName}}DSBools{{short .Kind}}(a []{{asType .Kind}}, b {{asType .Kind}}) (retVal []bool) {
	retVal = make([]bool, len(a))
	for i, v := range a {
		retVal[i] = v {{.OpSymb}} b
	}
	return retVal
}
`

var (
	genericVecVecCmpSame    *template.Template
	genericVecVecCmpBool    *template.Template
	genericVecScalarCmpSame *template.Template
	genericVecScalarCmpBool *template.Template
)

func init() {
	genericVecVecCmpSame = template.Must(template.New("vvCmpSame").Funcs(funcs).Parse(genericVecVecCmpSameRaw))
	genericVecVecCmpBool = template.Must(template.New("vvCmpBool").Funcs(funcs).Parse(genericVecVecCmpBoolRaw))

	genericVecScalarCmpSame = template.Must(template.New("vsCmpSame").Funcs(funcs).Parse(genericVecScalarCmpSameRaw))
	genericVecScalarCmpBool = template.Must(template.New("vsCmpBool").Funcs(funcs).Parse(genericVecScalarCmpBoolRaw))
}

func genericCmp(f io.Writer, generic *ManyKinds) {
	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		for _, k := range generic.Kinds {
			if bo.is(k) {
				op := ArithBinOp{k, bo.OpName, bo.OpSymb, false}
				genericVecVecCmpBool.Execute(f, op)
				fmt.Fprint(f, "\n")
				if isNumber(k) {
					genericVecVecCmpSame.Execute(f, op)
					fmt.Fprint(f, "\n")
				}

				genericVecScalarCmpBool.Execute(f, op)
				fmt.Fprint(f, "\n")
				if isNumber(k) {
					genericVecScalarCmpSame.Execute(f, op)
					fmt.Fprint(f, "\n")
				}
			}
			fmt.Fprintln(f, "\n")
		}
	}
}
