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

		if reuse.len() != a.len() {
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

		if reuse.len() != a.len() {
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
	switch {
	case incr:
		// when incr returned, the reuse is the *Dense to be incremented
		switch reuse.t.Kind() {
		{{range .Kinds -}}
			{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			data := reuse.{{asType . | strip}}s()
			for i := range data {
				{{if or $div $scaleInv -}}
					{{if hasPrefix .String "float" -}}
					{{else if hasPrefix .String "complex" -}}
					{{else -}}
					if other.get{{short .}}(i) == 0 {
						errs = append(errs, i)
						continue
					}
					{{end -}}
				{{end -}}
				data[i] += {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "float32"}}
							math32.Pow(t.get{{short .}}(i), other.get{{short .}}(i))
						{{else if eq .String "float64" -}}
							math.Pow(t.get{{short .}}(i), other.get{{short .}}(i))
						{{else if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(t.get{{short .}}(i)), complex128(other.get{{short .}}(i))))
						{{else if eq .String "complex128" -}}
							cmplx.Pow(t.get{{short .}}(i), other.get{{short .}}(i))
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
		copyDense(reuse, t)
		reuse.{{lower .OpName}}(other)
		retVal = reuse
	case safe:
		retVal = recycledDense(t.t, t.shape.Clone())
		copyDense(retVal, t)
		retVal.{{lower .OpName}}(other)
	case !safe:
		t.{{lower .OpName}}(other)
		retVal = t
	}
	return
}
`

const denseDenseArithSwitchTableRaw = `func (t *Dense) {{lower .OpName}}(other *Dense){
	switch t.t.Kind() {
	{{$op := .OpName -}}
	{{range .Kinds -}}
		{{if isNumber . -}}
	case reflect.{{reflectKind .}}:
		{{lower $op}}{{short .}}(t.{{asType . | strip}}s(), other.{{asType . | strip}}s())
		{{end -}}
	{{end -}}
	}
}
`

const denseScalarArithRaw = `func (t *Dense) {{.OpName}}(other interface{}, opts ...FuncOpt) (retVal *Dense, err error){
	reuse, safe, toReuse, incr, err := prepUnaryDense(t, opts...)
	if err != nil {
		return nil, err
	}

	{{$isFunc := .IsFunc -}}
	{{$op := .OpSymb -}}
	{{$scaleInv := hasPrefix .OpName "ScaleInv" -}}
	{{$div := hasPrefix .OpName "Div" -}}
	{{if or $scaleInv $div -}}var errs errorIndices
	{{end -}}
	switch {
	case incr:
		switch t.t.Kind(){
		{{range .Kinds -}}
		{{if isNumber . -}}
		case reflect.{{reflectKind .}}:
			data := reuse.{{asType . | strip}}s()
			b := other.({{asType .}})
			for i := range data {
				{{if or $div $scaleInv -}}
					{{if hasPrefix .String "float" -}}
					{{else if hasPrefix .String "complex" -}}
					{{else -}}
					if b == 0 {
						errs = append(errs, i)
						continue
					}
					{{end -}}
				{{end -}}
				data[i] += {{if $isFunc -}}
					{{if eq $op "math.Pow" -}}
						{{if eq .String "float32"}}
							math32.Pow(t.get{{short .}}(i), b)
						{{else if eq .String "float64" -}}
							math.Pow(t.get{{short .}}(i), b)
						{{else if eq .String "complex64" -}}
							complex64(cmplx.Pow(complex128(t.get{{short .}}(i)), complex128(b)))
						{{else if eq .String "complex128" -}}
							cmplx.Pow(t.get{{short .}}(i), b)
						{{else -}}
							{{asType .}}({{$op}}(float64(t.get{{short .}}(i)), float64(b)))
						{{end -}}
					{{end -}}
				{{else -}}
					t.get{{short .}}(i) {{$op}} b
				{{end -}}
		}
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
		{{lower $op}}{{short .}}(t.{{asType . | strip}}s(), b)
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
