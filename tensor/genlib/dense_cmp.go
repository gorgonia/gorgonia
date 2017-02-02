package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

var cmpBinOps = []struct {
	OpName string
	OpSymb string
	is     func(reflect.Kind) bool
}{
	{"Eq", "==", isEq},
	{"Gt", ">", isOrd},
	{"Gte", ">=", isOrd},
	{"Lt", "<", isOrd},
	{"Lte", "<=", isOrd},
}

const prepCmpRaw = `func prepBinaryDenseCmp(a, b *Dense, opts ...FuncOpt)(reuse *Dense, safe, same, toReuse bool, err error) {
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(dtypeMismatch, a.t, b.t)
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return 
	}
	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	if !safe{
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)
		if same {
			if reuse.t.Kind() != a.t.Kind() {
				err = errors.Errorf(dtypeMismatch, a.t, reuse.t)
				return
			}	
		} else {
			if reuse.t.Kind() != reflect.Bool {
				err = errors.Errorf(dtypeMismatch, reflect.Bool, reuse.t)
				return
			}
		}

		if  err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}
`

const eleqordDDRaw = `func (t *Dense) {{lower .OpName}}DD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	{{$op := .OpName}}
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{ $eq := isEq . -}}
		{{ $ord := isOrd . -}}
		{{ $opEq := eq $op "Eq" -}}
		{{ $eeq := and $eq $opEq}}

		{{if or $eeq $ord -}}
	case reflect.{{reflectKind .}}:
		td := t.{{sliceOf .}}
		od := other.{{sliceOf .}}
			{{if isNumber . -}}
				if same {
					ret := {{lower $op}}DDSame{{short .}}(td, od)
					retVal.fromSlice(ret)
				} else {
					ret := {{lower $op}}DDBools{{short .}}(td, od)
					retVal.fromSlice(ret)
				}
			{{else -}}
				ret := {{lower $op}}DDBools{{short .}}(td, od)
				retVal.fromSlice(ret)
			{{end -}}

		{{end -}}

	{{end -}}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "{{lower .OpName}}")
		return
	}

	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

`

var (
	ddElEqOrd *template.Template
)

func init() {
	ddElEqOrd = template.Must(template.New("ElEqOrdDD").Funcs(funcs).Parse(eleqordDDRaw))

}

func denseCmp(f io.Writer, generic *ManyKinds) {
	fmt.Fprintln(f, prepCmpRaw)
	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{generic, bo.OpName, bo.OpSymb, false}
		ddElEqOrd.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}

}
