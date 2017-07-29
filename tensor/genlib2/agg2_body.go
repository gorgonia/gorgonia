package main

import "text/template"

// level 2 aggregation (tensor.StdEng) templates

const prepVVRaw = `var reuse *Dense
	var safe, toReuse, incr {{if ne .TypeClassCheck "Number"}}, same{{end}} bool
	if reuse, safe, toReuse, {{if eq .TypeClassCheck "Number"}}incr, _,{{else}}_, same,{{end}} err = prepBinaryTensor(a, b, {{.TypeClassCheck | lower}}Types, opts...); err != nil {
		return
	}

	if reuse != nil && !reuse.IsNativelyAccessible() {
		err = errors.Errorf(inaccessibleData, reuse)
		return
	}

	typ := a.Dtype().Type
	var dataA, dataB, dataReuse *storage.Header
	var ait, bit, iit Iterator
	var useIter bool
	if dataA, dataB, dataReuse, ait, bit, iit, useIter, err = prepDataVV(a, b, reuse); err != nil {
		err = errors.Wrapf(err, "StdEng.{{.Name}}")
		return
	}
`

const prepMixedRaw = `var reuse *Dense
	var safe, toReuse, incr {{if ne .TypeClassCheck "Number"}}, same{{end}} bool
	if reuse, safe, toReuse, {{if eq .TypeClassCheck "Number"}}incr, _,{{else}}_, same,{{end}} err = prepUnaryTensor(t, {{.TypeClassCheck | lower}}Types, opts...); err != nil {
		return
	}

	a := t
	typ := t.Dtype().Type
	var ait, bit,  iit Iterator
	var dataA, dataB, dataReuse *storage.Header
	var useIter bool

	if leftTensor {
		if dataA, dataB, dataReuse, ait, iit, useIter, err = prepDataVS(t, s, reuse); err != nil {
			err = errors.Wrapf(err, "StdEng.{{.Name}}")
			return
		}
	} else {
		if dataA, dataB, dataReuse, bit, iit, useIter, err = prepDataSV(s, t, reuse); err != nil {
			err = errors.Wrapf(err, "StdEng.{{.Name}}")
			return
		}	
	}

`

const agg2BodyRaw = `if useIter {
		switch {
		case incr:
			err = e.E.{{.Name}}IterIncr(typ, dataA, dataB, dataReuse, ait, bit, iit)
			retVal = reuse
		case toReuse:
			storage.Copy(dataReuse, dataA, typ)
			err = e.E.{{.Name}}Iter(typ, dataReuse, dataB, ait, bit)
			retVal = reuse
		case !safe:
			err = e.E.{{.Name}}Iter(typ, dataA, dataB, ait, bit)
			retVal = a
		default:
			ret := a.Clone().(headerer)
			err = e.E.{{.Name}}Iter(typ, ret.hdr(), dataB, ait, bit)
			retVal = ret.(Tensor)
		}
		return
	}
	switch {
	case incr:
		err = e.E.{{.Name}}Incr(typ, dataA, dataB, dataReuse)
		retVal = reuse
	case toReuse:
		storage.Copy(dataReuse, dataA, typ)
		err = e.E.{{.Name}}(typ, dataReuse, dataB)
		retVal = reuse
	case !safe:
		err = e.E.{{.Name}}(typ, dataA, dataB)
		retVal = a
	default:
		ret := a.Clone().(headerer)
		err = e.E.{{.Name}}(typ, ret.hdr(), dataB)
		retVal = ret.(Tensor)
	}
	return
`

const agg2CmpBodyRaw = `
	if !same && !toReuse{
		reuse = NewDense(Bool, a.Shape().Clone(), WithEngine(e))
		dataReuse = reuse.array.hdr()
		iit = IteratorFromDense(reuse)
	}

	if useIter {
		switch {
			case !toReuse && same:
				err = e.E.{{.Name}}SameIter(typ, dataA, dataB, ait, bit)
				retVal = a
			case toReuse && same:
				storage.Copy(dataReuse,dataA, typ)
				err = e.E.{{.Name}}SameIter(typ, dataReuse, dataB, ait, bit)
				retVal = reuse
			case toReuse && !same:
				err = e.E.{{.Name}}Iter(typ, dataA, dataB, dataReuse, ait, bit, iit)
				retVal = reuse
			case !safe:
				err = e.E.{{.Name}}SameIter(typ, dataA, dataB, ait, bit)
				retVal = a
			default:
				err = e.E.{{.Name}}Iter(typ, dataA, dataB, dataReuse, ait, bit, iit)
				retVal = reuse
		}
		return
	}
	switch {
			case !toReuse && same:
				err = e.E.{{.Name}}Same(typ, dataA, dataB)
				retVal = a
			case toReuse && same:
				storage.Copy(dataReuse,dataA, typ)
				err = e.E.{{.Name}}Same(typ, dataReuse, dataB)
				retVal = reuse
			case toReuse && !same:
				err = e.E.{{.Name}}(typ, dataA, dataB, dataReuse)
				retVal = reuse
			case !safe:
				err = e.E.{{.Name}}Same(typ, dataA, dataB)
				retVal = a
			default:
				err = e.E.{{.Name}}(typ, dataA, dataB, dataReuse)
				retVal = reuse
	}
	return
`

var (
	prepVV      *template.Template
	prepMixed   *template.Template
	agg2Body    *template.Template
	agg2CmpBody *template.Template
)

func init() {
	prepVV = template.Must(template.New("prepVV").Funcs(funcs).Parse(prepVVRaw))
	prepMixed = template.Must(template.New("prepMixed").Funcs(funcs).Parse(prepMixedRaw))
	agg2Body = template.Must(template.New("agg2body").Funcs(funcs).Parse(agg2BodyRaw))
	agg2CmpBody = template.Must(template.New("agg2CmpBody").Funcs(funcs).Parse(agg2CmpBodyRaw))
}
