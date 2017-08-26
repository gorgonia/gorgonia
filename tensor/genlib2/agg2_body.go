package main

import "text/template"

// level 2 aggregation (tensor.StdEng) templates

const cmpPrepRaw = `var safe, same bool
	if reuse, safe, _, _, same, err = handleFuncOpts({{.VecVar}}.Shape(), {{.VecVar}}.Dtype(), false, opts...); err != nil{
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}
	if !safe {
		same = true
	}
`

const arithPrepRaw = `var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts({{.VecVar}}.Shape(), {{.VecVar}}.Dtype(), true, opts...); err != nil{
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}	
`

const prepVVRaw = `if err = binaryCheck(a, b, {{.TypeClassCheck | lower}}Types); err != nil {
		return nil, errors.Wrapf(err, "{{.Name}} failed")
	}

	var reuse DenseTensor
	{{template "prep" . -}}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		return nil, errors.Errorf(inaccessibleData, reuse)
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter, swap bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, swap, err = prepDataVV(a, b, reuse); err != nil {
		return nil, errors.Wrapf(err, "StdEng.{{.Name}}")
	}
`

const prepMixedRaw = `if err = unaryCheck(t, {{.TypeClassCheck | lower}}Types); err != nil {
		return nil, errors.Wrapf(err, "{{.Name}} failed")
	}

	var reuse DenseTensor
	{{template "prep" . -}}

	a := t
	typ := t.Dtype().Type
	var ait, bit,  iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.{{.Name}}")
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			return nil, errors.Wrapf(err, opFail, "StdEng.{{.Name}}")
		}	
	}

`

const prepUnaryRaw = `if err = unaryCheck(a, {{.TypeClassCheck | lower}}Types); err != nil {
		err = errors.Wrapf(err, "{{.Name}} failed")
		return
	}
	var reuse DenseTensor
	var safe, toReuse, incr bool
	if reuse, safe, toReuse, incr, _, err = handleFuncOpts(a.Shape(), a.Dtype(), true, opts...); err != nil {
		return nil, errors.Wrap(err, "Unable to handle funcOpts")
	}

	typ := a.Dtype().Type
	var ait, rit Iterator
	var dataA, dataReuse *storage.Header
	var useIter bool

	if dataA, dataReuse, ait, rit, useIter, err = prepDataUnary(a, reuse); err != nil{
		return nil, errors.Wrapf(err, opFail, "StdEng.{{.Name}}")
	}
	`

const agg2BodyRaw = `if useIter {
		switch {
		case incr:
			err = e.E.{{.Name}}IterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		{{if .VV -}}	
		case toReuse:
			storage.CopyIter(typ,dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.{{.Name}}Iter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		{{else -}}
		case toReuse && leftTensor:
			storage.CopyIter(typ, dataReuse, dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.{{.Name}}Iter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		case toReuse && !leftTensor:
			storage.CopyIter(typ, dataReuse, dataB, iit, bit)
			iit.Reset()
			bit.Reset()
			err = e.E.{{.Name}}Iter(typ, dataA, dataReuse, ait, iit)
			retVal = reuse
		{{end -}}
		case !safe:
			err = e.E.{{.Name}}Iter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
		{{if .VV -}}
			var ret headerer
			if swap{
				ret = b.Clone().(headerer)
			}else{
				ret = a.Clone().(headerer)
			}
			err = e.E.{{.Name}}Iter(typ, ret.hdr(), dataB, ait, bit)
		{{else -}}
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.{{.Name}}Iter(typ, ret.hdr(), dataB, ait, bit)
			}else {
				err = e.E.{{.Name}}Iter(typ, dataA, ret.hdr(), ait, bit)
			}
		{{end -}}
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.{{.Name}}Incr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	{{if .VV -}}
	case toReuse:
		storage.Copy(typ,dataReuse, dataA)
		err = e.E.{{.Name}}(typ, dataReuse, dataB)
		retVal = reuse
	{{else -}}
	case toReuse && leftTensor:
		storage.Copy(typ, dataReuse, dataA)
		err = e.E.{{.Name}}(typ, dataReuse, dataB)
		retVal = reuse
	case toReuse && !leftTensor:
		storage.Copy(typ, dataReuse, dataB)
		err = e.E.{{.Name}}(typ, dataA, dataReuse)
		retVal = reuse
	{{end -}}
	case !safe:
		err = e.E.{{.Name}}(typ, dataA, dataB)
		retVal = a
	default:
		{{if .VV -}}
			var ret headerer
			if swap {
				ret = b.Clone().(headerer)
			}else{
				ret = a.Clone().(headerer)
			}
			err = e.E.{{.Name}}(typ, ret.hdr(), dataB)
		{{else -}}
			ret := a.Clone().(headerer)
			if leftTensor {
				err = e.E.{{.Name}}(typ, ret.hdr(), dataB)
			}else {
				err = e.E.{{.Name}}(typ, dataA, ret.hdr())	
			}
		{{end -}}
		retVal = ret.(Tensor)
	}
	return
`

const agg2CmpBodyRaw = `// check to see if anything needs to be created
	switch {
	case same && safe && reuse == nil:
		{{if .VV -}}
		if swap{
			reuse = NewDense(b.Dtype(), b.Shape().Clone(), WithEngine(e))
		} else{
			reuse = NewDense(a.Dtype(), a.Shape().Clone(), WithEngine(e))
		}
		{{else -}}
		reuse = NewDense(a.Dtype(), a.Shape().Clone(), WithEngine(e))
		{{end -}}
		dataReuse = reuse.hdr()
		iit = IteratorFromDense(reuse)
	case !same && safe && reuse == nil:
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse =  reuse.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
		case !safe && same && reuse == nil:
			err = e.E.{{.Name}}SameIter(typ, dataA, dataB, ait, bit)
			retVal = a
		case same && safe && reuse != nil:
			storage.CopyIter(typ,dataReuse,dataA, iit, ait)
			ait.Reset()
			iit.Reset()
			err = e.E.{{.Name}}SameIter(typ, dataReuse, dataB, iit, bit)
			retVal = reuse
		default: // safe && bool
			err = e.E.{{.Name}}Iter(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		}
		return
	}
	switch {
		case !safe && same && reuse == nil:
			err = e.E.{{.Name}}Same(typ, dataA, dataB)
			retVal = a
		case same && safe && reuse != nil:
			storage.Copy(typ,dataReuse,dataA)
			err = e.E.{{.Name}}Same(typ, dataReuse, dataB)
			retVal = reuse
		default:
			err = e.E.{{.Name}}(typ, dataA, dataB, dataReuse)
			retVal = reuse
	}
	return
`

const agg2UnaryBodyRaw = `
	if useIter{
		switch {
		case incr:
			cloned:= a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok {
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.{{.Name}}Iter(typ, h.hdr(), ait); err != nil {
				return nil, errors.Wrap(err, "Unable to perform {{.Name}}")
			}
			ait.Reset()
			err = e.E.AddIter(typ, dataReuse, h.hdr(), rit, ait)
			retVal = reuse
		case toReuse:
			storage.CopyIter(typ, dataReuse, dataA, rit, ait)
			rit.Reset()
			err = e.E.{{.Name}}Iter(typ, dataReuse, rit)
			retVal = reuse
		case !safe:
			err = e.E.{{.Name}}Iter(typ, dataA, ait)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok{
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.{{.Name}}Iter(typ, h.hdr(), ait)
			retVal = cloned
		}
		return
	}
		switch {
		case incr:
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok{
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			if err = e.E.{{.Name}}(typ, h.hdr()); err != nil {
				return nil, errors.Wrap(err, "Unable to perform {{.Name}}")
			}
			err = e.E.Add(typ, dataReuse, h.hdr())
			retVal = reuse
		case toReuse:
			storage.Copy(typ, dataReuse, dataA)
			err = e.E.{{.Name}}(typ, dataReuse)
			retVal = reuse
		case !safe:
			err = e.E.{{.Name}}(typ, dataA)
			retVal = a
		default: // safe by default
			cloned := a.Clone().(Tensor)
			h, ok := cloned.(headerer)
			if !ok{
				return nil, errors.Errorf("Unable to clone a %T - not a headerer", a)
			}
			err = e.E.{{.Name}}(typ, h.hdr())
			retVal = cloned
		}
		return
`

var (
	prepVV        *template.Template
	prepMixed     *template.Template
	prepUnary     *template.Template
	agg2Body      *template.Template
	agg2CmpBody   *template.Template
	agg2UnaryBody *template.Template
)

func init() {
	prepVV = template.Must(template.New("prepVV").Funcs(funcs).Parse(prepVVRaw))
	prepMixed = template.Must(template.New("prepMixed").Funcs(funcs).Parse(prepMixedRaw))
	prepUnary = template.Must(template.New("prepUnary").Funcs(funcs).Parse(prepUnaryRaw))
	agg2Body = template.Must(template.New("agg2body").Funcs(funcs).Parse(agg2BodyRaw))
	agg2CmpBody = template.Must(template.New("agg2CmpBody").Funcs(funcs).Parse(agg2CmpBodyRaw))
	agg2UnaryBody = template.Must(template.New("agg2UnaryBody").Funcs(funcs).Parse(agg2UnaryBodyRaw))
}
