package main

import (
	"fmt"
	"io"
	"text/template"
)

var engSubRaw = ` // Sub performs subtraction: a -= b. 
// If the element type are numbers, it will fall to using standard number subtraction.
// If the element type are strings, concatenation will happen instead.
// If the element type is unknown, it will attempt to look for the Suber interface, and use the Suber method.
// Failing that, it produces an error.
//
// a and b may be scalars. In the case where one of which is vector and the other is a scalar, it's the vector that will
// be clobbered. If both are scalars, a will be clobbered
func (e E) Sub(t reflect.Type, a, b Array) (err error) {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		
		switch {
		case as && bs:
			at[0] -= bt[0]
		case as && !bs:
			sub{{short .}}VS(bt, at[0])
		case bs && !as:
			sub{{short .}}VS(at, bt[0])
		default:
			sub{{short .}}(at, bt)
		}
		return nil
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
}

// SubIter performs subtraction a -= b, guided by the iterator
func (e E) SubIter(t reflect.Type, a, b Array, ait, bit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}

		switch {
		case as && bs :
			at[0] -= bt[0]
			return nil
		case as && !bs:
			return sub{{short .}}IterVS(bt, at[0], bit)
		case bs && !as:
			return sub{{short .}}IterVS(at, bt[0], ait)
		default:
			return sub{{short .}}Iter(at, bt, ait, bit)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
}

// SubIncr performs incr += a - b 
func (e E) SubIncr(t reflect.Type, a, b, incr Array) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is{
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}		

		switch {
		case as && bs :
			if !is {
				at[0]-=bt[0]
				return e.Add(t, incr, a)
			}
			it[0] += at[0]-bt[0]
			return nil
		case as && !bs:
			subIncr{{short .}}VS(bt, at[0], it)
		case bs && !as:
			subIncr{{short .}}VS(at, bt[0], it)
		default:
			subIncr{{short .}}(at, bt, it)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

// SubIterIncr performs incr += a - b, guided by iterators
func (e E) SubIterIncr(t reflect.Type, a, b, incr Array, ait, bit, iit Iterator) error {
	as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is{
		return errors.Errorf("Cannot increment on a scalar increment")
	}

	switch t {
		{{range .Kinds -}}
		{{if isNumber . -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}

		switch {
		case as && bs:
			if !is {
				at[0] -= bt[0]
				return e.SubIter(t, incr, a, iit, ait)
			}
		case as && !bs:
			return subIncr{{short .}}IterVS(bt, at[0], it, bit, iit)
		case bs && !as:
			return subIncr{{short .}}IterVS(at, bt[0], it, ait, iit)
		default:
			return subIncr{{short .}}Iter(at, bt, it, ait, bit, iit)
		}
		{{end -}}
		{{end -}}
	default:
		return errors.Errorf("NYI")
	}
	return nil
}

`

var stdengGenericSubRaw = `
func sub{{short .}}(a, b []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}Sub(a, b)
	{{else -}}
		a = a[:len(a)]
		b = b[:len(a)]
		for i, v := range b {
			a[i] -= v
		}
	{{end -}}
}

func sub{{short .}}VS(a []{{asType .}}, b {{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}TransInv(a, b)
	{{else -}}
		a = a[:len(a)]
		for i := range a {
			a[i] -= b
		}
	{{end -}}
}

func sub{{short .}}SV(a {{asType .}}, b []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}TransInvR(b, a)
	{{else -}}
		for i := range b {
			b[i] = a - b[i]
		}
	{{end -}}
}

func sub{{short .}}Iter(a, b []{{asType .}}, ait, bit Iterator) (err error) {	
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		a[i] -= b[j]
	}
	return
}

func sub{{short .}}IterVS(a []{{asType .}}, b {{asType .}}, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		a[i] -= b
	}
	err = handleNoOp(err)
	return
}

func sub{{short .}}IterSV(a {{asType .}}, b []{{asType .}}, it Iterator) (err error) {
	var i int
	for i, _, err = it.NextValid(); err == nil; i, _, err = it.NextValid() {
		b[i] = a - b[i]
	}
	err = handleNoOp(err)
	return	
}

func subIncr{{short .}}(a, b, incr []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}IncrSub(a, b, incr)
	{{else -}}
		a = a[:len(a)]
		b = b[:len(a)]
		incr = incr[:len(a)]
		for i, v := range a {
			incr[i] += v - b[i]
		}
	{{end -}}
}

func subIncr{{short .}}VS(a []{{asType .}}, b {{asType .}}, incr []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}IncrTransInv(a, b, incr)
	{{else -}}
		a = a[:len(a)]
		incr = incr[:len(a)]
		for i := range a {
			incr[i] += a[i] - b
		}
	{{end -}}
}

func subIncr{{short .}}SV(a {{asType .}}, b []{{asType .}}, incr []{{asType .}}) {
	{{if isFloat . -}}
		{{vecPkg .}}IncrTransInvR(b, a, incr)
	{{else -}}
		b = b[:len(b)]
		incr = incr[:len(b)]
		for i := range b {
			incr[i] += a - b[i]
		}
	{{end -}}
}

func subIncr{{short .}}IterVS(a []{{asType .}}, b {{asType .}}, incr []{{asType .}}, ait, iit Iterator)(err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a[i] - b
	}
	return nil
}

func subIncr{{short .}}IterSV(a {{asType .}}, b []{{asType .}}, incr []{{asType .}}, ait, iit Iterator)(err error) {
	var i, j int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[j] += a - b[i]
	}
	return nil
}

func subIncr{{short .}}Iter(a, b, incr []{{asType .}}, ait, bit, iit Iterator) (err error) {
	var i, j, k int
	for {
		if i, _, err = ait.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if j, _, err = bit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, _, err = iit.NextValid(); err != nil {
			err = handleNoOp(err)
			break
		}
		incr[k] += a[i] - b[j]
	}
	return nil
}
`

var (
	stdengGenericSub *template.Template
	engSub           *template.Template
)

func init() {
	stdengGenericSub = template.Must(template.New("stdeng generic sub").Funcs(funcs).Parse(stdengGenericSubRaw))
	engSub = template.Must(template.New("e.Sub").Funcs(funcs).Parse(engSubRaw))
}

func generateStdEngSub(f io.Writer, generic *ManyKinds) {
	engSub.Execute(f, generic)
	fmt.Fprint(f, "\n")
	for _, k := range generic.Kinds {
		if isNumber(k) {
			stdengGenericSub.Execute(f, k)
			fmt.Fprint(f, "\n")
		}
	}
}
