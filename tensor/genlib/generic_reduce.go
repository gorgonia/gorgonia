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
		v := reflect.ValueOf(l.Get(i))
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
const genericSliceMinMaxRaw = `func sliceMin{{short .}}(a []{{asType .}}) {{asType .}}{
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduce{{short .}}(min{{short .}}, a[0], a[1:]...)
}

func sliceMax{{short .}}(a []{{asType .}}) {{asType .}}{
	if len(a) < 1 {
		panic("Max of empty slice is meaningless")
	}
	return reduce{{short .}}(max{{short .}}, a[0], a[1:]...)
}

`

var (
	genericReduce      *template.Template
	genericSum         *template.Template
	genericSliceMinMax *template.Template
)

func init() {
	genericReduce = template.Must(template.New("genericReduce").Funcs(funcs).Parse(genericReduceRaw))
	genericSum = template.Must(template.New("genericSum").Funcs(funcs).Parse(genericSumRaw))
	genericSliceMinMax = template.Must(template.New("genericSliceMinMax").Funcs(funcs).Parse(genericSliceMinMaxRaw))
}

func genericReduction(f io.Writer, generic *ManyKinds) {
	fmt.Fprintln(f, reflectBasedReduceRaw)
	for _, k := range generic.Kinds {
		if !isParameterized(k) {
			genericReduce.Execute(f, k)
		}
	}

	for _, k := range filter(generic.Kinds, isNumber) {
		genericSum.Execute(f, k)
	}

	for _, k := range filter(generic.Kinds, isOrd) {
		if isNumber(k) {
			genericSliceMinMax.Execute(f, k)
		}
	}
}
