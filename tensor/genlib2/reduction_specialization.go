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
	Typeclass   TypeClass
}

var reductionOps = []ReductionOp{
	{OpName: "Sum", VecVec: "VecAdd", OpOfVec: "Sum", GenericName: "Add", Typeclass: isNumber},
	{OpName: "Max", VecVec: "VecMax", OpOfVec: "SliceMax", GenericName: "Max", Typeclass: isNonComplexNumber},
	{OpName: "Min", VecVec: "VecMin", OpOfVec: "SliceMin", GenericName: "Min", Typeclass: isNonComplexNumber},
}

const reductionSpecializationRaw = `func Monotonic{{.OpName | title}}(t reflect.Type, a *storage.Header) (retVal interface{}, err error) {
	switch t {
		{{$opOfVec := .OpOfVec -}}
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		retVal = {{$opOfVec}}{{short .}}(a.{{sliceOf .}})
		return
		{{end -}}
		{{end -}}
	default:
		err = errors.Errorf("Cannot perform {{.OpName}} on %v", t)
		return
	}
}

func {{.OpName | title}}Methods(t reflect.Type)(firstFn, lasFn, defaultFn interface{}, err error) {
	{{$vecVec := .VecVec -}}
	{{$opOfVec := .OpOfVec -}}
	{{$genericName := .GenericName -}}
	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		return {{$vecVec}}{{short .}}, {{$opOfVec}}{{short .}}, {{$genericName}}{{short .}}, nil
		{{end -}}
		{{end -}}
	default:
		return nil, nil, nil, errors.Errorf("No methods found for {{.OpName}} for %v", t)
	}
}

`

var reductionSpecialization *template.Template

func init() {
	reductionSpecialization = template.Must(template.New("reduction specialization").Funcs(funcs).Parse(reductionSpecializationRaw))
}

func generateReductionSpecialization(f io.Writer, ak Kinds) {
	for _, op := range reductionOps {
		for _, k := range ak.Kinds {
			if !op.Typeclass(k) {
				continue
			}
			op.Kinds = append(op.Kinds, k)
		}
		reductionSpecialization.Execute(f, op)
	}
}
