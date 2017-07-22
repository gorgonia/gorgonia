package main

import "text/template"

// level 1 aggregation (internal.E) templates

const (
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

	eMapRaw = `as := isScalar(a)
	switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		var f0 {{template "fntype0" .}}
		var f1 {{template "fntype1" .}}

		switch f := fn.(type){
		case {{template "fntype0" .}}:
			f0 = f
		case {{template "fntype1" .}}:
			f1 = f
		default:
			return errors.Errorf("Cannot map fn of %T to array", fn)
		}

		at := a.{{sliceOf .}}
		switch{
		case as && incr && f0 != nil:
			at[0] += f0(at[0])
		case as && incr && f0 == nil:
			var tmp {{asType .}}
			if tmp, err= f1(at[0]); err != nil {
				return
			}
			at[0] += tmp
		case as && !incr && f0 != nil:
			at[0], err = f1(at[0])
		case as && !incr && f0 == nil:
			at[0], err = f1(at[0])
		case !as && incr && f0 != nil:
			MapIncr{{short .}}(f0, at)
		case !as && incr && f0 == nil:
			err = MapIncrErr{{short .}}(f1, at)
		case !as && !incr && f0 == nil:
			err = MapErr{{short .}}(f1, at)
		default:
			Map{{short .}}(f0, at)
		}
		{{end -}}
	default:
		return errors.Errorf("Cannot map t of %v", t)
	
	}
	`

	eMapIterRaw = `switch t {
		{{range .Kinds -}}
	case {{reflectKind .}}:
		at := a.{{sliceOf .}}
		var f0 {{template "fntype0" .}}
		var f1 {{template "fntype1" .}}

		switch f := fn.(type){
		case {{template "fntype0" .}}:
			f0 = f
		case {{template "fntype1" .}}:
			f1 = f
		default:
			return errors.Errorf("Cannot map fn of %T to array", fn)
		}

		switch {
		case incr && f0 != nil:
			MapIncrIter{{short .}}(f0, at, ait)
		case incr && f0 == nil:
			err = MapIncrIterErr{{short .}}(f1, at, ait)
		case !incr && f0 == nil:
			err = MapIterErr{{short .}}(f1, at, ait)
		default:
			MapIter{{short .}}(f0, at, ait)
		}
		{{end -}}
	default:
			return errors.Errorf("Cannot map t of %v", t)
	}
	`
)

var (
	eArith         *template.Template
	eArithIncr     *template.Template
	eArithIter     *template.Template
	eArithIncrIter *template.Template
	eMap           *template.Template
	eMapIter       *template.Template
)

func init() {
	eArith = template.Must(template.New("eArith").Funcs(funcs).Parse(eArithRaw))
	eArithIncr = template.Must(template.New("eArithIncr").Funcs(funcs).Parse(eArithIncrRaw))
	eArithIter = template.Must(template.New("eArithIter").Funcs(funcs).Parse(eArithIterRaw))
	eArithIncrIter = template.Must(template.New("eArithIncrIter").Funcs(funcs).Parse(eArithIncrIterRaw))

	eMap = template.Must(template.New("eMap").Funcs(funcs).Parse(eMapRaw))
	eMapIter = template.Must(template.New("eMapIter").Funcs(funcs).Parse(eMapIterRaw))
}
