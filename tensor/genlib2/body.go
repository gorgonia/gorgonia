package main

import "text/template"

type LoopBody struct {
	TypedBinOp
	Range    string
	Left     string
	Right    string
	IterName string
}

const (
	vvLoopRaw = `for i := range {{.Range}} {
		{{template "check" . -}}
		{{if .IsFunc -}}
			{{.Range}}[i] = {{ template "callFunc" . }}
		{{else -}}
			{{.Range}}[i] = {{.Left}} {{template "symbol" .Kind}} {{.Right}}
		{{end -}}
	}`

	vvIncrLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}
		{{if .IsFunc -}}
			{{.Range}}[i] += {{template "callFunc" .}}
		{{else -}}
			{{.Range}}[i] += {{.Left}} {{template "symbol" .Kind}} {{.Right}}
		{{end -}}
	}`

	vvIterLoopRaw = `var i, j int
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
		if validi && validj {
			{{template "check" . -}}

			{{if .IsFunc -}}
				{{.Range}}[i] = {{template "callFunc" .}}
			{{else -}}
				{{.Range}}[i] = {{.Left}} {{template "symbol" .Kind}} {{.Right}}
			{{end -}}		
		}
	}`

	vvIterIncrLoopRaw = `var i, j, k int
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
		if validi && validj && validk {
			{{template "check" . -}}
			{{if .IsFunc -}}
				{{.Range}}[k] += {{template "callFunc" .}}
			{{else -}}
				{{.Range}}[k] += {{.Left}} {{template "symbol" .Kind}}  {{.Right}}
			{{end -}}		
		}
	}`

	mixedLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}

		{{if .IsFunc -}}
			{{.Range}}[i] = {{template "callFunc" .}}
		{{else -}}
			{{.Range}}[i]  {{template "symbol" .Kind}}=  {{.Right}}
		{{end -}}
	}`

	mixedIncrLoopRaw = `for i := range {{.Range}}{
		{{template "check" . -}}

		{{if .IsFunc -}}
			{{.Range}}[i] += {{template "callFunc" .}}
		{{else -}}
			{{.Range}}[i]  += {{.Left}} {{template "symbol" .Kind}}  {{.Right}}
		{{end -}}	
	}`

	mixedIterLoopRaw = `var i int
	var validi bool
	for {
		if i, validi, err = {{.IterName}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi {
			{{template "check" . -}}

			{{if .IsFunc -}}
				{{.Range}}[i] = {{template "callFunc" .}}
			{{else -}}
				{{.Range}}[i] = {{.Left}} {{template "symbol" .Kind}} {{.Right}}
			{{end -}}
		}
	}`

	mixedIterIncrLoopRaw = `var i, k int
	var validi, validk bool
	for {
		if i, validi, err = {{.IterName}}.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if k, validk, err = iit.NextValidity(); err != nil {
			err = handleNoOp(err)
			break
		}
		if validi && validk {
			{{template "check" . -}}
			{{if .IsFunc -}}
				{{.Range}}[k] += {{template "callFunc" .}}
			{{else -}}
				{{.Range}}[k] += {{.Left}} {{template "symbol" .Kind}}  {{.Right}}
			{{end -}}		
		}
	}
	`

	// did I mention how much I hate C-style macros? Now I'm doing them instead
	callFunc = `{{if eq "complex64" .Kind.String -}}
		complex64({{template "symbol" .Kind}}(complex128({{.Left}}), complex128({{.Right}})))
		{{else -}}
		{{template "symbol" .Kind}}({{.Left}}, {{.Right}})
		{{end -}}
	`
	check0 = `if {{.Right}} == 0 {
		errs = append(errs, i)
		{{.Range}}[i] = 0
		continue
	}
	`

	eArithRaw = `as := isScalar(a)
	bs := isScalar(b)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			{{$isDiv := eq $name "Div" -}}
			{{$p := panicsDiv0 . -}}
			switch {
			case as && bs:
				at[0] += bt[0]
			case as && !bs:
				{{if and $isDiv $p}} err = {{end}} {{$name}}SV{{short .}}(at[0], bt)
			case !as && bs:
				{{if and $isDiv $p}} err = {{end}} {{$name}}VS{{short .}}(at, bt[0])
			default:
				{{if and $isDiv $p}} err = {{end}} {{$name}}{{short .}}(at, bt)
			}
			return 
		{{end -}}
		default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eArithIncrRaw = `as := isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)
	if (as && !bs) || (bs && !as) && is {
		return errors.Errorf("Cannot increment on scalar increment. a: %d, b %d", a.Len(), b.Len())
	}
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
		case {{reflectKind .}}:
			at := a.{{sliceOf .}}
			bt := b.{{sliceOf .}}
			it := incr.{{sliceOf .}}

			switch {
			case as && bs:
				{{$name}}{{short .}}(at, bt)
				if !is {
					return e.Add(t, incr, a)
				}
				it[0]+= at[0]
			case as && !bs:
				{{$name}}IncrSV{{short .}}(at[0], bt, it)
			case !as && bs :
				{{$name}}IncrVS{{short .}}(at, bt[0], it)
			default:
				{{$name}}Incr{{short .}}(at, bt,it)
			}
			return 
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`
	eArithIterRaw = `as := isScalar(a)
	bs := isScalar(b)
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		switch {
		case as && bs :
			{{$name}}{{short .}}(at, bt)
		case as && !bs:
			{{$name}}IterSV{{short .}}(at[0], bt, bit)
		case !as && bs:
			{{$name}}IterVS{{short .}}(at, bt[0], ait)
		default:
			{{$name}}Iter{{short .}}(at, bt, ait, bit)
		}
		return
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`

	eArithIncrIterRaw = `as :=isScalar(a)
	bs := isScalar(b)
	is := isScalar(incr)

	if (as && !bs) || (bs && !as) && is {
		return errors.Errorf("Cannot increment on a scalar increment. len(a): %d, len(b) %d", a.Len(), b.Len())
	}
	{{$name := .Name}}
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		bt := b.{{sliceOf .}}
		it := incr.{{sliceOf .}}
		switch {
		case as && bs:
			{{$name}}{{short .}}(at, bt)
			if !is {
				return e.{{$name}}Iter{{short .}}(t, incr, a, iit, ait)
			}
			it[0] += at[0]
			return
		case as && !bs:
			return {{$name}}IncrIterSV{{short .}}(at[0], bt, it, bit, iit)
		case !as && bs:
			return {{$name}}IncrIterVS{{short .}}(at, bt[0], it, ait, iit)
		default:
			return {{$name}}IncrIter{{short .}}(at, bt, it, ait, bit, iit)
		}
		{{end -}}
	default:
		return errors.Errorf("Unsupported type %v for {{$name}}", t)
	}
	`
)

var (
	eArith         *template.Template
	eArithIncr     *template.Template
	eArithIter     *template.Template
	eArithIncrIter *template.Template
)

func init() {
	eArith = template.Must(template.New("eArith").Funcs(funcs).Parse(eArithRaw))
	eArithIncr = template.Must(template.New("eArithIncr").Funcs(funcs).Parse(eArithIncrRaw))
	eArithIter = template.Must(template.New("eArithIter").Funcs(funcs).Parse(eArithIterRaw))
	eArithIncrIter = template.Must(template.New("eArithIncrIter").Funcs(funcs).Parse(eArithIncrIterRaw))
}
