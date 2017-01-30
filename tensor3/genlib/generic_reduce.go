package main

import (
	"fmt"
	"io"
	"text/template"
)

const reflectBasedReduceRaw = `func reduceRef(f reflect.Value, fnT reflect.Type, def reflect.Value, l *Dense) interface{} {
	retVal := def
	if l.len() == 0 {
		return retVal.Interface()
	}

	args := make([]reflect.Value, 0, fnT.NumIn())
	for i := 0; i < l.len(); i++ {
		v := reflect.ValueOf(l.get(i))
		args = append(args, retVal)
		args = append(args, v)
		retVal = f.Call(args)[0]
		args = args[:0]
	}
	return retVal.Interface()
}

`
const genericReduceRaw = `func reduce{{short .}}(f func(a, b {{asType .}}) {{asType .}}, def {{asType .}}, l ...{{asType .}}) (retVal {{asType .}}){
	retVal = def
	if len(l) == 0 {
		return 
	}

	for _, v := range l {
		retVal = f(retVal, v)
	}
	return
}

`

const genericSumRaw = `func sum{{short .}}(a []{{asType .}}) {{asType .}}{ return reduce{{short .}}(add{{short .}}, 0, a...)}
`

var (
	genericReduce *template.Template
	genericSum    *template.Template
)

func init() {
	genericReduce = template.Must(template.New("genericReduce").Funcs(funcs).Parse(genericReduceRaw))
	genericSum = template.Must(template.New("genericSum").Funcs(funcs).Parse(genericSumRaw))
}

func genericReduction(f io.Writer, generic *ManyKinds) {
	fmt.Fprintln(f, reflectBasedReduceRaw)
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			genericReduce.Execute(f, k)
		}
	}

	for _, k := range generic.Kinds {
		if isNumber(k) {
			genericSum.Execute(f, k)

		}
	}
}
