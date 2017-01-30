package main

import (
	"io"
	"reflect"
	"text/template"
)

type ReductionOp struct {
	OpName      string
	VecVec      string // sum(a, b []T)
	OpOfVec     string // sum([]T)
	GenericName string // sum(T, T) T
	Kinds       []reflect.Kind
}

var reductionOps = []ReductionOp{
	{OpName: "sum", VecVec: "vecAdd", OpOfVec: "sum", GenericName: "add"},
}

const denseReductionMethodsRaw = ` // {{title .OpName}} returns the {{.OpName}} of the elements of the tensor along the given axes.
// If multiple axes are given then this method will return the {{.OpName}} of the Tensor according the the order of the axes provided
func (t *Dense) {{title .OpName}}(along ...int)(retVal *Dense, err error) {
	monotonic, incr1 := IsMonotonicInts(along) // if both are true, then it means all axes are accounted for, then it'll return a scalar value
	if (monotonic && incr1 && len(along) == t.Dims()) || len(along) == 0 {
		{{$opOfVec := .OpOfVec -}}
		var ret interface{}
		switch t.t.Kind() {
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			ret = {{$opOfVec}}{{short .}}(t.{{sliceOf .}})
		{{end -}}
		{{end -}}
		}
		retVal = New(FromScalar(ret))
		return
	}
	retVal = t
	prev := -1
	dims := len(retVal.Shape())

	for _, axis := range along {
		if prev == -1 {
			prev = axis
		}
		if axis > prev {
			axis--
		}

		if axis >= dims {
			err = errors.Errorf(dimMismatch, retVal.Dims(), axis)
			return
		}

		retVal = retVal.{{.OpName}}(axis)
	}
	return
}
func (t *Dense) {{.OpName}}(axis int)(retVal *Dense){
	{{$vecvec := .VecVec -}}
	{{$generic := .GenericName -}}
	switch t.t.Kind(){
	{{range .Kinds -}}
	{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		return t.sReduce{{short .}}(axis, {{$vecvec}}{{short .}}, {{$opOfVec}}{{short .}}, {{$generic}}{{short .}})
	{{end -}}
	{{end -}}
	}
	panic("Unreachable")
}
`

var (
	denseReductionMethods *template.Template
)

func init() {
	denseReductionMethods = template.Must(template.New("denseReductionMethods").Funcs(funcs).Parse(denseReductionMethodsRaw))
}

func generateDenseReductionMethods(f io.Writer, generic *ManyKinds) {
	for _, op := range reductionOps {
		op.Kinds = generic.Kinds
		denseReductionMethods.Execute(f, op)
	}
}
