package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

var maskcmpMethods = []struct {
	Name     string
	Desc     string
	NumArgs  int
	CmpFn    string
	ReqFloat bool
	Kinds    []reflect.Kind
}{
	{"MaskedEqual", "equal to ", 1, "a == x", false, nil},
	{"MaskedNotEqual", "not equal to ", 1, "a != x", false, nil},
	{"MaskedValues", " equal to ", 3, "math.Abs(float64(a-x)) <= delta", true, nil},
	{"MaskedGreater", " greater than ", 1, "a > x", false, nil},
	{"MaskedGreaterEqual", " greater than or equal to ", 1, "a >= x", false, nil},
	{"MaskedLess", " less than ", 1, "a < x", false, nil},
	{"MaskedLessEqual", " less than or equal to ", 1, "a <= x", false, nil},
	{"MaskedInside", " inside range of ", 2, "(a >= x) && (a <= y)", false, nil},
	{"MaskedOutside", " outside range of ", 2, "(a < x) || (a > y)", false, nil},
}

const maskCmpMethodRaw = `// {{.Name}} sets the mask to true where the corresponding data is {{.Desc}} val
// Any values must be the same type as the tensor
func (t *Dense) {{.Name}}({{if ge .NumArgs 1 -}} val1 interface{} {{end}} {{if ge .NumArgs 2 -}} , val2 interface{} {{end}}  {{if ge .NumArgs 3 -}} , val3 ...interface{}{{end}})(err error){
	{{if .ReqFloat}}
	if !isFloat(t.t) {
			err = errors.Errorf("Can only do {{.Name}} with floating point types")
			return
	}
	{{end}}
	
	if !t.IsMasked() {
		t.SetMaskStrides(t.strides)
		t.fix()
	}	
    it := MultIteratorFromDense(t)
	runtime.SetFinalizer(it, destroyMultIterator)
	
    {{$numargs := .NumArgs}}
	{{$name := .Name}}
    {{$fn := .CmpFn}}	
	{{$reqFloat := .ReqFloat}}	
	switch t.t.Kind(){
	{{range .Kinds -}}
	{{if isParameterized . -}}
	{{else -}}
	{{if or (not (isOrd .)) (and $reqFloat (isntFloat .)) -}}
	{{else -}}
		case reflect.{{reflectKind .}}:
			data := t.{{sliceOf .}}
			mask := t.mask
			{{if ge $numargs 1 -}} x := val1.({{asType .}}) {{end}}
			{{if ge $numargs 2 -}} y := val2.({{asType .}}){{end}}
			{{if ge $numargs 3 -}} 
				{{if eq $name "MaskedValues"}} 
					delta := float64(1.0e-8)
					if len(val3) > 0 {
					delta = float64(val3[0].({{asType .}})) + float64(y)*math.Abs(float64(x))
					}
				{{else}}
					z := val3.({{asType .}})
				{{end}}
			{{end}}			
			for i, err := it.Next(); err == nil; i, err = it.Next() {
				j := it.LastMaskIndex(0)
				a := data[i]
				mask[j] = mask[j] || ({{$fn}})
			}
			it.Reset()
			
    {{end}}
    {{end}}
	{{end}}
}
return nil
}
`

var (
	maskCmpMethod *template.Template
)

func init() {
	maskCmpMethod = template.Must(template.New("maskcmpmethod").Funcs(funcs).Parse(maskCmpMethodRaw))
}

func maskcmpmethods(f io.Writer, generic *ManyKinds) {
	for _, mm := range maskcmpMethods {
		mm.Kinds = generic.Kinds
		fmt.Fprintf(f, "/* %s */ \n\n", mm.Name)
		maskCmpMethod.Execute(f, mm)

	}
}
