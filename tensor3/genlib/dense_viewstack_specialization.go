package main

import (
	"fmt"
	"io"
	"text/template"
)

const doviewstackSpecialRaw = `func (t *Dense) doViewStack{{short .}}(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int){
	data := retVal.{{asType . | strip}}s()[:0]
	for i := 0; i < batches; i++ {
		for j := 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.{{asType . | strip}}s()[id])
		}
		for j, ot := range others {
			for k := 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.{{asType . | strip}}s()[id])
			}
		}
	}
}
`
const doviewstackRaw = `func (t *Dense) doViewStack(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int){
	switch t.t.Kind(){
	{{range .Kinds -}}
		{{if isSpecialized . -}}
	case reflect.{{reflectKind .}}:
		t.doViewStack{{short .}}(retVal, axisStride, batches, ch, others, chs)
		{{end -}}
	{{end -}}
	default:
		var index int
		for i := 0; i < batches; i++ {
			for j := 0; j < axisStride; j++ {
				id, ok := <-ch
				if !ok {
					break
				}
				retVal.set(index, t.get(id))
				index++
			}
			for j, ot := range others {
				for k := 0; k < axisStride; k++ {
					id, ok := <-chs[j]
					if !ok {
						break
					}
					retVal.set(index, ot.get(id))
					index++
				}
			}
		}
	}
}
`

var (
	DoViewStackSpecialization *template.Template
	DoViewStack               *template.Template
)

func init() {
	DoViewStackSpecialization = template.Must(template.New("doViewStackSpec").Funcs(funcs).Parse(doviewstackSpecialRaw))
	DoViewStack = template.Must(template.New("doViewStack").Funcs(funcs).Parse(doviewstackRaw))
}

func viewstack(f io.Writer, generic *ManyKinds) {
	for _, k := range generic.Kinds {
		if isSpecialized(k) {
			fmt.Fprintf(f, "/* %v */\n\n", k)
			DoViewStackSpecialization.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
	DoViewStack.Execute(f, generic)
	fmt.Fprintf(f, "\n\n\n")
}
