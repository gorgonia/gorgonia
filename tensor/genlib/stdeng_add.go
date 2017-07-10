package main

import (
	"fmt"
	"io"
	"text/template"
)

var engAddRaw = ` // Add performs addition: a += b. 
// If the element type are numbers, it will fall to using standard number addition.
// If the element type are strings, concatenation will happen instead.
// If the element type is unknown, it will attempt to look for the Adder interface, and use the Adder method.
// Failing that, it produces an error.
//
// a and b may be scalars. In the case where one of which is vector and the other is a scalar, it's the vector that will
// be clobbered. If both are scalars, a will be clobbered
func (e E) Add(t reflect.Type, a, b Array) (err error) {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
		{{range .Kinds -}}
		{{if isAddable . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		
		switch {
		case as && bs:
			at[0] += bt[0]
		case as && !bs:
			add{{short .}}VS(bt, at[0])
		case bs && !as:
			add{{short .}}VS(at, bt[0])
		default:
			add{{short .}}(at, bt)
		}
		return nil
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
}

// AddIter performs addition a += b, guided by the iterator
func (e E) AddIter(t reflect.Type, a, b Array, ait, bit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
		{{range .Kinds -}}
		{{if isAddable . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}

		switch {
		case as && bs :
			at[0] += bt[0]
			return nil
		case as && !bs:
			return add{{short .}}IterVS(bt, at[0], bit)
		case bs && !as:
			return add{{short .}}IterVS(at, bt[0], ait)
		default:
			return add{{short .}}Iter(at, bt, ait, bit)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
}

// AddIncr performs incr += a + b 
func (e E) AddIncr(t reflect.Type, a, b, incr Array) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is{
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
		{{range .Kinds -}}
		{{if isAddable . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}		

		switch {
		case as && bs :
			if !is {
				at[0]+=bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0]+bt[0]
			return nil
		case as && !bs:
			addIncr{{short .}}VS(bt, at[0], it)
		case bs && !as:
			addIncr{{short .}}VS(at, bt[0], it)
		default:
			addIncr{{short .}}(at, bt, it)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

// AddIterIncr performs incr += a + b, guided by iterators
func (e E) AddIterIncr(t reflect.Type, a, b, incr Array, ait, bit, iit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is{
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
		{{range .Kinds -}}
		{{if isAddable . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}

		switch {
		case as && bs:
			if !is {
				at[0] += bt[0]
				return e.AddIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return addIncr{{short .}}IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return addIncr{{short .}}IterVS(at, bt[0], it, ait, iit)
		default:
			return addIncr{{short .}}Iter(at, bt, it, ait, bit, iit)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

`

var stdengGenericAddRaw = `
func add{{short .}}(a, b []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}Add(a, b)
	{{else -}}
		a = a[:len(a)]
		b = b[:len(a)]
		for i, v := range b {
			a[i] += v
		}
	{{end -}}
}

func add{{short .}}VS(a []{{asType .}}, b {{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}Trans(a, b)
	{{else -}}
		a = a[:len(a)]
		for i := range a {
			a[i] += b
		}
	{{end -}}
}

func add{{short .}}Iter(a, b []{{asType .}}, ait, bit Iterator) (err error) {	
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}

		if (validi && validj) {
			a[i] += b[j]
		}

	}
	return
}

func add{{short .}}IterVS(a []{{asType .}}, b {{asType .}}, it Iterator) (err error) {
	var i int
	var valid bool
	for i, valid, err = it.NextValidity(); err == nil; i, valid, err = it.NextValidity() {
		if valid {
			a[i] += b
		}
	}
	err = handleNoOp(err)
	return
}

func addIncr{{short .}}(a, b, incr []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}IncrAdd(a, b, incr)
	{{else -}}
		a = a[:len(a)]
		b = b[:len(a)]
		incr = incr[:len(a)]
		for i, v := range a {
			incr[i] += v + b[i]
		}
	{{end -}}
}

func addIncr{{short .}}VS(a []{{asType .}}, b {{asType .}}, incr []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}IncrScale(a, b, incr)
	{{else -}}
		a = a[:len(a)]
		incr = incr[:len(a)]
		for i := range a {
			incr[i] += a[i] + b
		}
	{{end -}}
}

func addIncr{{short .}}IterVS(a []{{asType .}}, b {{asType .}}, incr []{{asType .}}, ait, iit Iterator)(err error) {
	var i, j int
	var validi, validj bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = iit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if (validi && validj) {
			incr[j] += a[i] + b
		}
	}
	return nil
}

func addIncr{{short .}}Iter(a, b, incr []{{asType .}}, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	var validi, validj, validk bool
	for {
		if i, validi, err = ait.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, validj, err = bit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, validk, err = iit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if (validi && validj && validk){
			incr[k] += a[i]+b[j]
		}
	}
	return nil
}
`

var (
	stdengGenericAdd *template.Template
	engAdd           *template.Template
)

func init() {
	stdengGenericAdd = template.Must(template.New("stdeng generic add").Funcs(funcs).Parse(stdengGenericAddRaw))
	engAdd = template.Must(template.New("e.Add").Funcs(funcs).Parse(engAddRaw))
}

func generateStdEngAdd(f io.Writer, generic *ManyKinds) {
	engAdd.Execute(f, generic)
	fmt.Fprint(f, "\n")
	for _, k := range generic.Kinds {
		if isAddable(k) {
			stdengGenericAdd.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
