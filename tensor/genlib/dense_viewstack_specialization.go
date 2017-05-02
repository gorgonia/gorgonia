package main

import (
	"fmt"
	"io"
	"text/template"
)

const doviewstackSpecialRaw = `func (t *Dense) doViewStack{{short .}}(retVal *Dense, axisStride, batches int, ch chan int, others []*Dense, chs []chan int){
	data := retVal.{{sliceOf .}}[:0]
	mask := retVal.mask[:0]
		if t.IsMasked(){
		fmt.Println("do this")
		}
	retIsMasked:= t.IsMasked()
	for _, ot := range others {
		retIsMasked = retIsMasked || ot.IsMasked()
		}
	for i := 0; i < batches; i++ {
		isMasked := t.IsMasked()
		var j int
		for j = 0; j < axisStride; j++ {
			id, ok := <-ch
			if !ok {
				break
			}
			data = append(data, t.{{sliceOf .}}[id])
			if isMasked {
				mask = append(mask, t.mask[id])
				}
		}
		if retIsMasked && (!isMasked) {
			mask = append(mask, make([]bool,j)...)
		}

		var ot *Dense
		for j, ot = range others {
			isMasked = ot.IsMasked()
			var k int
			for k = 0; k < axisStride; k++ {
				id, ok := <-chs[j]
				if !ok {
					break
				}
				data = append(data, ot.{{sliceOf .}}[id])
				if isMasked {
					mask = append(mask, ot.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool,k)...)
			}
		}
	}
	retVal.mask=mask
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
		retIsMasked:= t.IsMasked()
		mask := retVal.mask[:0]
		for _, ot := range others {
			retIsMasked = retIsMasked || ot.IsMasked()
			}
		for i := 0; i < batches; i++ {
			isMasked := t.IsMasked()
			var j int
			for j = 0; j < axisStride; j++ {
				id, ok := <-ch
				if !ok {
					break
				}
				retVal.Set(index, t.Get(id))
				index++
				if isMasked {
					mask = append(mask, t.mask[id])
				}
			}
			if retIsMasked && (!isMasked) {
				mask = append(mask, make([]bool,j)...)
			}
			var ot *Dense
			for j, ot = range others {
				isMasked = ot.IsMasked()
				var k int
				for k = 0; k < axisStride; k++ {
					id, ok := <-chs[j]
					if !ok {
						break
					}
					retVal.Set(index, ot.Get(id))
					index++
					if isMasked {
						mask = append(mask, ot.mask[id])
					}
				}
				if retIsMasked && (!isMasked) {
					mask = append(mask, make([]bool,k)...)
				}
			}
		}
		retVal.mask=mask
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
