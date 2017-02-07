package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type ArgMethod struct {
	ArgName string
	Kinds   []reflect.Kind
}

var argnames = []string{"Argmax", "Argmin"}

const argMethodRaw = `func (t *Dense) {{.ArgName}}(axis int)(retVal *Dense, err error){
	if axis == AllAxes {
		return t.{{lower .ArgName}}(nil)
	}

	if axis >= len(t.Shape()) {
		err = errors.Errorf(dimMismatch, len(t.Shape()), axis)
		return
	}

	
	axes := make([]int, len(t.Shape()))
	for i := range t.Shape() {
		switch {
		case i < axis:
			axes[i] = i
		case i == axis:
			axes[len(axes)-1] = i
		case i > axis:
			axes[i-1] = i
		}
	}

	// be a good citizen - borrow and return, since we're only using this AP to figure out the moves
	newAP, _, err := t.AP.T(axes...)
	if _, ok := err.(NoOpError); !ok && err != nil {
		return
	} else if ok {
		err = nil // reset errs
		newAP = t.AP.Clone()
	}
	defer ReturnAP(newAP)

	it := NewFlatIterator(newAP)
	return t.{{lower .ArgName}}(it)
}

func (t *Dense) {{lower .ArgName}}(it *FlatIterator) (retVal *Dense, err error) {
	var lastSize, next int
	var newShape Shape
	var indices []int
	if it != nil {
		lastSize = it.Shape()[len(it.Shape())-1]		
		newShape = it.Shape().Clone()
		newShape = newShape[:len(newShape)-1]
		defer ReturnInts(newShape)
	}
	{{$arg := .ArgName}}
	switch t.t.Kind() {
		{{range .Kinds -}}
		{{if isNumber .}}
		{{if hasPrefix .String "complex" -}}
		{{else -}}
		case reflect.{{reflectKind .}}:
			if it == nil {
				retVal = New(FromScalar({{lower $arg}}{{short .}}(t.{{sliceOf .}})))
				return
			}
			data := t.{{asType . | strip }}s()
			tmp := make([]{{asType .}}, 0, lastSize)
			for next, err = it.Next(); err == nil; next, err = it.Next() {
				tmp = append(tmp, data[next])

				if len(tmp) == lastSize {
					am := {{lower $arg}}{{short .}}(tmp)
					indices = append(indices, am)

					// reset
					tmp = tmp[:0]
				}
			}
			if _, ok := err.(NoOpError); !ok && err != nil {
				return
			}
			err = nil
			retVal = New(WithShape(newShape...), WithBacking(indices))
			return
		{{end -}}
		{{end -}}
		{{end -}}
	}
	panic("Unreachable")
}

`

var (
	argMethod *template.Template
)

func init() {
	argMethod = template.Must(template.New("argmethod").Funcs(funcs).Parse(argMethodRaw))
}

func argmethods(f io.Writer, generic *ManyKinds) {
	for _, an := range argnames {
		fmt.Fprintf(f, "/* %s */ \n\n", an)
		op := ArgMethod{an, generic.Kinds}
		argMethod.Execute(f, op)
	}
}
