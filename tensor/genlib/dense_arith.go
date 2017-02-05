package main

import (
	"fmt"
	"io"
	"reflect"
	"text/template"
)

type ArithBinOp struct {
	Kind   reflect.Kind
	OpName string
	OpSymb string
	IsFunc bool
}

type BinOps struct {
	*ManyKinds
	OpName string
	OpSymb string
	IsFunc bool
}

type ArithBinOps struct {
	BinOps

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}

var binOps = []struct {
	OpName string
	OpSymb string

	IsFunc bool

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}{
	{"Add", "+", false, true, 0, true, true, false, "", ""},
	{"Sub", "-", false, true, 0, false, false, false, "Add", "+"},
	{"Mul", "*", false, true, 1, true, true, false, "", ""},
	{"Div", "/", false, true, 1, false, false, false, "Mul", "*"},
	{"Pow", "math.Pow", true, false, 1, false, false, false, "", ""},
}

var vecscalarOps = []struct {
	OpName string
	OpSymb string

	IsFunc bool

	HasIdentity   bool
	Identity      int
	IsCommutative bool
	IsAssociative bool
	IsInv         bool
	InvOpName     string
	InvOpSymb     string
}{
	{"Trans", "+", false, true, 0, true, true, false, "", ""},
	{"TransInv", "-", false, true, 0, false, false, false, "Trans", "+"},
	{"TransInvR", "-", false, false, 0, false, false, false, "", ""},
	{"Scale", "*", false, true, 1, true, true, false, "", ""}, // no good way to test inverse w/o  implicit broadcast
	{"ScaleInv", "/", false, true, 1, false, false, false, "Scale", "*"},
	{"ScaleInvR", "/", false, false, 0, false, false, false, "", ""},    // identity is 0 on purpose
	{"PowOf", "math.Pow", true, false, 0, false, false, false, "", ""},  // identity is 0 on purpose
	{"PowOfR", "math.Pow", true, false, 0, false, false, false, "", ""}, // identity is 0 on purpose
}

const prepArithRaw = `func prepBinaryDense(a, b *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t) && !isNumber(b.t) {
		err = noopError{}
		return
	}
	if a.t.Kind() != b.t.Kind() {
		err = errors.Errorf(typeMismatch, a.t, b.t)
		return
	}
	if !a.Shape().Eq(b.Shape()) {
		err = errors.Errorf(shapeMismatch, b.Shape(), a.Shape())
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}

		if reuse.t.Kind() != a.t.Kind(){
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return 
		}

		if reuse.len() != a.Shape().TotalSize() {
			err =  errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}

func prepUnaryDense(a *Dense, opts ...FuncOpt) (reuse *Dense, safe, toReuse, incr bool, err error) {
	if !isNumber(a.t){
		err = noopError{}
		return
	}

	fo := parseFuncOpts(opts...)
	reuseT, incr := fo.incrReuse()
	safe = fo.safe()
	toReuse = reuseT != nil

	if toReuse {
		if reuse, err = getDense(reuseT); err != nil {
			err = errors.Wrapf(err, "Cannot reuse a different type of Tensor in a *Dense-Scalar operation")
			return
		}
		

		if reuse.t.Kind() != a.t.Kind(){
			err = errors.Errorf(typeMismatch, a.t, reuse.t)
			err = errors.Wrapf(err, "Cannot use reuse")
			return 
		}

		if reuse.len() != a.Shape().TotalSize() {
			err =  errors.Errorf(shapeMismatch, reuse.Shape(), a.Shape())
			err = errors.Wrapf(err, "Cannot use reuse: shape mismatch")
			return
		}
	}
	return
}
`

const denseDenseArithRaw = `func (t *Dense) {{.OpName}}(other *Dense, opts ...FuncOpt) (retVal *Dense, err error){
	reuse, safe, toReuse, incr, err := prepBinaryDense(t, other, opts...)
	if err != nil {
		return nil, err
	}

	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{if or $scaleInv $div -}}var errs errorIndices
	{{end -}}
	var it, ot *FlatIterator
	if t.IsMaterializable() {
		it = NewFlatIterator(t.AP)
	}
	if other.IsMaterializable() {
		ot = NewFlatIterator(other.AP)
	}
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		switch reuse.t.Kind() {
		{{range .Kinds -}}
			{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			data := reuse.{{sliceOf .}}
			for i := range data {
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if other.get{{short .}}(i) == 0 {
						errs = append(errs, i)
						continue
					}
					{{end -}}
				{{end -}}
				data[i] += {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(t.getC64(i)), complex128(other.getC64(i))))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(t.get{{short .}}(i), other.get{{short .}}(i))
						{{else -}}
							{{asType .}}({{$op}}(float64(t.get{{short .}}(i)), float64(other.get{{short .}}(i))))
						{{end -}}
					{{end -}}
				{{else -}}
					t.get{{short .}}(i) {{$op}} other.get{{short .}}(i)
				{{end -}}
			}
			{{end -}}
		{{end -}}
		}
		{{if or $scaleInv $div -}}
		if errs != nil {
			err = err
		}
		{{end -}}
		retVal = reuse
	case toReuse:
		if t.IsMaterializable(){
			copyDenseIter(reuse, t, nil, it)
		} else {
			copyDense(reuse, t) // technically copyDenseIter would have done the same but it's much slower
		}
		err = reuse.{{lower .OpName}}(other, nil, ot)
		retVal = reuse
	case safe:
		if t.IsMaterializable(){
			retVal = t.Materialize().(*Dense)
		} else {
			retVal = t.Clone().(*Dense)
		}
		err = retVal.{{lower .OpName}}(other, nil, ot)
	case !safe:
		err = t.{{lower .OpName}}(other, it, ot)
		retVal = t
	}
	return
}
`

const denseDenseArithSwitchTableRaw = `func (t *Dense) {{lower .OpName}}(other *Dense, it, ot *FlatIterator) (err error){
	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$opName := .OpName -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{if or $scaleInv $div -}}var errs errorIndices
	{{end -}}
	switch t.t.Kind() {
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		tdata := t.{{sliceOf .}}
		odata := other.{{sliceOf .}}
		var i, j int
		switch {
		case it != nil && ot != nil:
			for {
				if i, err = it.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				if j, err = ot.Next(); err != nil {
					if _, ok := err.(NoOpError); !ok {
						return
					}
					err = nil
					break
				}
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
			}
		case it != nil && ot == nil:
			for i, err = it.Next(); err == nil; i, err = it.Next(){
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
				j++
			}
			if _, ok := err.(NoOpError); !ok {
				return
			}
			err = nil
		case it == nil && ot != nil:
			for j, err = ot.Next(); err == nil; j, err = ot.Next() {
				{{if or $div $scaleInv -}}
					{{if isntFloat . -}}
					if odata[j] == 0 {
						errs = append(errs, j)
						continue
					}
					{{end -}}
				{{end -}}
				tdata[i] = {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(tdata[i]), complex128(odata[j])))
						{{else if isFloat . -}}
							{{mathPkg .}}Pow(tdata[i], odata[j])
						{{else -}}
							{{asType .}}({{$op}}(float64(tdata[i]), float64(odata[j])))
						{{end -}}
					{{end -}}
				{{else -}}
					tdata[i] {{$op}} odata[j]
				{{end -}}
				i++
			}
		default:
			vec{{$opName}}{{short .}}(tdata, odata)
		}
		{{end -}}
	{{end -}}
	default:
		// TODO: Handle Number interface
	}
	{{if or $scaleInv $div -}}
		if errs != nil {
			err = err
		}
	{{end -}}
	return nil
}
`

const denseScalarArithRaw = `func (t *Dense) {{.OpName}}(other interface{}, opts ...FuncOpt) (retVal *Dense, err error){
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$opName := .OpName}}
	switch {
	case incr:
		switch t.t.Kind(){
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			err = incr{{$opName}}{{short .}}(t.{{sliceOf .}}, reuse.{{sliceOf .}}, other.({{asType .}}))
			retVal = reuse	
		{{end -}}
		{{end -}}
		}
	case toReuse:
		copyDense(reuse, t)
		reuse.{{lower .OpName}}(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.Shape().Clone())
		copyDense(retVal, t)
		retVal.{{lower .OpName}}(other)
	case !safe:
		t.{{lower .OpName}}(other)
		retVal = t
	}
	return
}
`

const denseScalarArithSwitchTableRaw = `func (t *Dense) {{lower .OpName}}(other interface{}){
	switch t.t.Kind() {
	{{$op := .OpName -}}
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		b := other.({{asType .}})
		{{lower $op}}{{short .}}(t.{{sliceOf .}}, b)
		{{end -}}
		
	{{end -}}
	}
}
`

var (
	ddArith      *template.Template
	ddArithTable *template.Template
	dsArith      *template.Template
	dsArithTable *template.Template
)

func init() {
	ddArith = template.Must(template.New("dense arith").Funcs(funcs).Parse(denseDenseArithRaw))
	ddArithTable = template.Must(template.New("dense arith table").Funcs(funcs).Parse(denseDenseArithSwitchTableRaw))
	dsArith = template.Must(template.New("dense-scalar arith").Funcs(funcs).Parse(denseScalarArithRaw))
	dsArithTable = template.Must(template.New("dense-scalar arith table").Funcs(funcs).Parse(denseScalarArithSwitchTableRaw))
}
func arith(f io.Writer, generic *ManyKinds) {
	fmt.Fprint(f, prepArithRaw)
	for _, bo := range binOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{
			ManyKinds: generic,
			OpName:    bo.OpName,
			OpSymb:    bo.OpSymb,
			IsFunc:    bo.IsFunc,
		}
		ddArith.Execute(f, op)
		ddArithTable.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}
	for _, bo := range vecscalarOps {
		fmt.Fprintf(f, "/* %s */\n\n", bo.OpName)
		op := BinOps{generic, bo.OpName, bo.OpSymb, bo.IsFunc}
		dsArith.Execute(f, op)
		dsArithTable.Execute(f, op)
		fmt.Fprintln(f, "\n")
	}
}
