package main

import (
	"fmt"
	"io"
	"text/template"
)

// enerates the transpose specializations
const transposeSpecializedRaw = `func (t *Dense) transpose{{short .}}(expStrides []int){
	axes := t.transposeWith
	size := t.len()

	// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp {{asType .}}
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.set{{short .}}(i, saved)
			{{if eq .String "bool" -}}
				saved = false
			{{else if eq .String "string" -}}
				saved = ""
			{{else -}}
				saved = 0
			{{end -}}
			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.get{{short .}}(i)
		t.set{{short .}}(i, saved)
		saved = tmp

		i = dest
	}
}
`

const transposeRaw = `func (t *Dense) transpose(expStrides []int){
	switch t.t.Kind(){
	{{range .Kinds -}}
		{{if isSpecialized . -}}
	case reflect.{{reflectKind .}}:
		t.transpose{{short .}}(expStrides)
		{{end -}}
	{{end -}}
	default:
			axes := t.transposeWith
	size := t.len()
		// first we'll create a bit-map to track which elements have been moved to their correct places
	track := NewBitMap(size)
	track.Set(0)
	track.Set(size - 1) // first and last element of a transposedon't change

	// // we start our iteration at 1, because transposing 0 does noting.
	var saved, tmp interface{}
	var i int
	for i = 1; ; {
		dest := t.transposeIndex(i, axes, expStrides)

		if track.IsSet(i) && track.IsSet(dest) {
			t.set(i, saved)
			saved = reflect.Zero(t.t.Type).Interface()

			for i < size && track.IsSet(i) {
				i++
			}

			if i >= size {
				break
			}
			continue
		}

		track.Set(i)
		tmp = t.get(i)
		t.set(i, saved)
		saved = tmp

		i = dest
	}
	}
}
`

var (
	TransposeSpecialization *template.Template
	Transpose               *template.Template
)

func init() {
	TransposeSpecialization = template.Must(template.New("TransposeSpec").Funcs(funcs).Parse(transposeSpecializedRaw))
	Transpose = template.Must(template.New("Transpose").Funcs(funcs).Parse(transposeRaw))
}

func transpose(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isSpecialized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			TransposeSpecialization.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
	Transpose.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
}
