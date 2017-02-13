package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

var cmpBinOps = []struct {
	OpName  string
	OpSymb  string
	is      func(reflect.Kind) bool
	Inverse string
}{
	{"Eq", "==", isEq, "Eq"},
	{"Ne", "!=", isEq, "Ne"},
	{"Gt", ">", isOrd, "Lt"},
	{"Gte", ">=", isOrd, "Lte"},
	{"Lt", "<", isOrd, "Gt"},
	{"Lte", "<=", isOrd, "Gte"},
}

const prepCmpRaw = `func prepBinaryDenseCmp(a, b *Dense, opts ...FuncOpt)(reuse *Dense, safe, same, toReuse bool, err error) {
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(dtypeMismatch, a.t, b.t)
		return
	}

	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, a.Shape(), b.Shape())
		return 
	}
	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	if !safe{
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)
		if same {
			if reuse.t.Kind() != a.t.Kind() {
				err = errors.Errorf(dtypeMismatch, a.t, reuse.t)
				return
			}	
		} else {
			if reuse.t.Kind() != reflect.Bool {
				err = errors.Errorf(dtypeMismatch, reflect.Bool, reuse.t)
				return
			}
		}

		if  err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}

func prepUnaryDenseCmp(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, same, toReuse bool, err error){
	fo := parseFuncOpts(opts...)
	reuseT, _ := fo.incrReuse()
	safe = fo.safe()
	same = fo.same
	if !safe{
		same = true
	}
	toReuse = reuseT != nil

	if toReuse {
		reuse = reuseT.(*Dense)
		if same {
			if reuse.t.Kind() != a.t.Kind() {
				err = errors.Errorf(dtypeMismatch, a.t, reuse.t)
				return
			}	
		} else {
			if reuse.t.Kind() != reflect.Bool {
				err = errors.Errorf(dtypeMismatch, reflect.Bool, reuse.t)
				return
			}
		}

		if  err = reuseDenseCheck(reuse, a); err != nil {
			err = errors.Wrap(err, "Cannot use reuse")
			return
		}
	}
	return
}
`

const eleqordDDRaw = `func (t *Dense) {{lower .OpName}}DD(other *Dense, opts ...FuncOpt) (retVal *Dense, err error) {
	reuse, safe, same, toReuse, err := prepBinaryDenseCmp(t, other, opts...)
	if err != nil {
		return nil, err
	}

	{{$opName := .OpName -}}
	{{$op := .OpSymb -}}
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{ $eq := isEq . -}}
		{{ $ord := isOrd . -}}
		{{ $opEq := eq $opName "Eq" -}}
		{{ $opNe := eq $opName "Ne" -}}
		{{ $opE := or $opEq $opNe}}

		{{ $eeq := and $eq $opE -}}

		{{if or $eeq $ord -}}
	case reflect.{{reflectKind .}}:
		td := t.{{sliceOf .}}
		od := other.{{sliceOf .}}
		var i, j, k int
		var ret interface{} // slice of some sort
		switch {
		case t.IsMaterializable() && other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			ot := NewFlatIterator(other.AP)
			{{if isNumber . -}}
				var bs []bool
				var ss []{{asType .}}
				if same {
					ss = make([]{{asType .}}, t.Shape().TotalSize())
					ret = ss
				} else{
					bs = make([]bool, t.Shape().TotalSize())
					ret = bs
				}

			{{else -}}
				bs := make([]bool, t.Shape().TotalSize())
				ret = bs
			{{end -}}
			for {
				if i, err = it.Next(); err != nil {
					err = handleNoOp(err)
					break
				}
				if j, err = ot.Next(); err != nil {
					err = handleNoOp(err)
					break	
				}
				{{if isNumber . -}}
					if same {
						if td[i] {{$op}} od[j] {
							ss[k] = 1
						}
					} else {
						bs[k] = td[i] {{$op}} od[j]
					}
				{{else -}}
					bs[k] = td[i] {{$op}} od[j]
				{{end -}}
				k++
			}
			err = handleNoOp(err)
		case t.IsMaterializable() && !other.IsMaterializable():
			it := NewFlatIterator(t.AP)
			{{if isNumber . -}}
				var bs []bool
				var ss []{{asType .}}
				if same {
					ss = make([]{{asType .}}, t.Shape().TotalSize())
					ret = ss
				} else{
					bs = make([]bool, t.Shape().TotalSize())
					ret = bs
				}

			{{else -}}
				bs := make([]bool, t.Shape().TotalSize())
				ret = bs
			{{end -}}
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				{{if isNumber . -}}
					if same {
						if td[i] {{$op}} od[j] {
							ss[k] = 1
						}
					} else {
						bs[k] = td[i] {{$op}} od[j]
					}
				{{else -}}
					bs[k] = td[i] {{$op}} od[j]
				{{end -}}
				j++
				k++	
			}
			err = handleNoOp(err)
		case !t.IsMaterializable() && other.IsMaterializable():
			ot := NewFlatIterator(t.AP)
			{{if isNumber . -}}
				var bs []bool
				var ss []{{asType .}}
				if same {
					ss = make([]{{asType .}}, t.Shape().TotalSize())
					ret = ss
				} else{
					bs = make([]bool, t.Shape().TotalSize())
					ret = bs
				}

			{{else -}}
				bs := make([]bool, t.Shape().TotalSize())
				ret = bs
			{{end -}}
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				{{if isNumber . -}}
					if same {
						if td[i] {{$op}} od[j] {
							ss[k] = 1
						}
					} else {
						bs[k] = td[i] {{$op}} od[j]
					}
				{{else -}}
					bs[k] = td[i] {{$op}} od[j]
				{{end -}}
				i++
				k++	
			}
			err = handleNoOp(err)
		default:
			{{if isNumber . -}}
				if same {
					ret = {{lower $opName}}DDSame{{short .}}(td, od)
				} else {
					ret = {{lower $opName}}DDBools{{short .}}(td, od)
				}
			{{else -}}
				ret = {{lower $opName}}DDBools{{short .}}(td, od)
			{{end -}}
		}
		retVal.fromSlice(ret)

		{{end -}}

	{{end -}}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "{{lower .OpName}}")
	}
	
	if err != nil{
		return
	}

	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}

`

const eleqordDSRaw = `func (t *Dense) {{lower .OpName}}DS(other interface{}, opts ...FuncOpt) (retVal *Dense, err error){
	reuse, safe, same, toReuse, err := prepUnaryDenseCmp(t, opts...)
	if err != nil {
		return nil, err
	}
	{{$opName := .OpName}}
	{{$opEq := eq $opName "Eq" -}}
	{{ $opNe := eq $opName "Ne" -}}
	{{ $opE := or $opEq $opNe}}
	{{$op := .OpSymb}}
	retVal = recycledDenseNoFix(t.t, t.Shape().Clone())
	switch t.t.Kind() {
	{{range .Kinds -}}
	{{ $eq := isEq . -}}
		{{ $ord := isOrd . -}}
		{{ $eeq := and $eq $opE -}}

		{{if or $eeq $ord -}}
	case reflect.{{reflectKind .}}:
		data := t.{{sliceOf .}}
		b := other.({{asType .}})
		var ret interface{} // slice of some sort
		switch {
		case t.IsMaterializable():
			it := NewFlatIterator(t.AP)
			var i, j int 
			{{if isNumber . -}}
				var bs []bool
				var ss []{{asType .}}
				if same {
					ss = make([]{{asType .}}, t.Shape().TotalSize())
					ret = ss
				} else{
					bs = make([]bool, t.Shape().TotalSize())
					ret = bs
				}

			{{else -}}
				bs := make([]bool, t.Shape().TotalSize())
				ret = bs
			{{end -}}
			for i, err = it.Next(); err == nil; i, err = it.Next() {
				{{if isNumber .}}
					if same {
						if data[i] {{$op}} b{
							ss[j] = 1
						}
					} else {
						bs[j] = data[i] {{$op}} b
					}
				{{else -}}
					bs[j] = data[i] {{$op}} b
				{{end -}}
				j++
			}
		default:
			{{if isNumber . -}}
				if same {
					ret = {{lower $opName}}DSSame{{short .}}(data, b)
				} else {
					ret = {{lower $opName}}DSBools{{short .}}(data, b)
				}
			{{else -}}
			ret = {{lower $opName}}DSBools{{short .}}(data, b)
			{{end -}}
		}
		retVal.fromSlice(ret)
		{{end -}}
	{{end -}}
	default:
		err = errors.Errorf(unsupportedDtype, t.t, "{{lower .OpName}}")
		return
	}
	retVal.fix()
	err = retVal.sanity()

	switch {
	case toReuse:
		copyDense(reuse, retVal)
		ReturnTensor(retVal)
		retVal = reuse
	case !safe:
		copyDense(t, retVal)
		ReturnTensor(retVal)
		retVal = t
	}
	return
}
`

var (
	ddElEqOrd *template.Template
	dsElEqOrd *template.Template
)

func init() {
	ddElEqOrd = template.Must(template.New("ElEqOrdDD").Funcs(funcs).Parse(eleqordDDRaw))
	dsElEqOrd = template.Must(template.New("ElEqOrdDS").Funcs(funcs).Parse(eleqordDSRaw))
}

func denseCmp(f io.Writer, generic *ManyKinds) {
	fmt.Fprintln(f, prepCmpRaw)
	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{
			ManyKinds: generic,
			OpName:    bo.OpName,
			OpSymb:    bo.OpSymb,
		}
		ddElEqOrd.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}

	for _, bo := range cmpBinOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{
			ManyKinds: generic,
			OpName:    bo.OpName,
			OpSymb:    bo.OpSymb,
		}
		dsElEqOrd.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}

}
